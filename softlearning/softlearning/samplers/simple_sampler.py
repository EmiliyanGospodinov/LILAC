from collections import defaultdict

import numpy as np
from flatten_dict import flatten, unflatten

from softlearning.models.utils import flatten_input_structure
from .base_sampler import BaseSampler


class SimpleSampler(BaseSampler):
    def __init__(self, latent_dim=0, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None

        self._latent_dim = latent_dim
        if self._latent_dim == 2:
            self._current_latent = 0.1 * np.array([1, 0])
        elif self._latent_dim > 2:
            self._current_latent = np.zeros(self._latent_dim)
        self._total_samples = 0

    def attach(self, alg):
        self.alg = alg

    @property
    def _policy_input(self):
        observation = {
            key: self._current_observation[key][None, ...]
            for key in self.policy.observation_keys
        }
        policy_inputs = flatten_input_structure({
            **observation,
            'env_latents': self._current_latent[None, ...],
        })
        return policy_inputs

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation,
                        episode,
                        info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'episodes': [episode],
            'infos': info,
        }

        return processed_observation

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action = self.policy.actions_np(self._policy_input)[0]

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            episode=self._n_episodes,
            info=info,
        )

        for key, value in flatten(processed_sample).items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = unflatten({
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            })

            self.pool.add_path({
                key: value
                for key, value in last_path.items()
                if key != 'infos'
            })

            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self.pool.terminate_episode()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0

            self._current_path = defaultdict(list)

            self._n_episodes += 1

            feed_dict = {self.alg._placeholders['cached_latents']: self.alg._cached_latents}
            latents = self._session.run('latent_prior/latent_priors:0', feed_dict=feed_dict)
            self._current_latent = latents[self._n_episodes]
        else:
            self._current_observation = next_observation

        return self._n_episodes, next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        # observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch(batch_size, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics
