from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flatten_dict import flatten

from softlearning.models.utils import flatten_input_structure
from .rl_algorithm import RLAlgorithm

from softlearning.models.feedforward import feedforward_model

def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            plotter=None,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,

            stochastic_encoder=False,
            anneal_beta=False,
            batch_tasks=False,
            recon_loss=False,
            clip_grad=False,
            clip_value=10.,
            encoder_size=(64,64),
            decoder_size=(64,64),
            encoder_lr=1e-4,
            decoder_lr=1e-3,
            prior_lr=1e-3,
            pretrain_iters=10000,
            prior_sigma=1e-1,
            beta=1e-6,
            task_batch_size=32,
            batch_size=8,
            state_size=3,

            save_full_state=False,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._state_size = state_size
        self._pretrain_iters = pretrain_iters
        self._encoder_size = encoder_size
        self._decoder_size = decoder_size
        self._beta = beta

        self._stochastic_encoder = stochastic_encoder
        self._anneal_beta = anneal_beta
        self._batch_tasks = batch_tasks
        self._recon_loss = recon_loss
        self._clip_grad = clip_grad

        if self._batch_tasks:
            self._task_batch_size = task_batch_size
            self._batch_size = batch_size

        if self._stochastic_encoder:
            self._prior_sigma = prior_sigma
            self._prior_sigma_sq = self._prior_sigma ** 2

        if self._clip_grad:
            self._clip_value = clip_value

        self._cached_latents = np.zeros((self._max_episodes, self._latent_dim))
        self._cache_counts = np.zeros(self._max_episodes)

        self._encoder_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=encoder_lr,
            name="encoder_optimizer")

        self._prior_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=prior_lr,
            name="prior_optimizer")

        if self._recon_loss:
            self._decoder_optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=decoder_lr,
                name="decoder_optimizer")

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize

        self._save_full_state = save_full_state

        self._build()

    def _build(self):
        super(SAC, self)._build()

        self._init_encoder_update()
        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()

    def _get_Q_target(self):
        observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._policy.observation_keys
        }
        policy_inputs = flatten_input_structure(
            {**observations, 'env_latents': self.next_latents})

        next_actions = self._policy.actions(policy_inputs)
        next_log_pis = self._policy.log_pis(policy_inputs, next_actions)

        next_Q_observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._Qs[0].observation_keys
        }
        next_Q_observations = flatten_input_structure(
            {**next_Q_observations, 'env_latents': self.next_latents})
        next_Q_inputs = flatten_input_structure(
            {'observations': next_Q_observations, 'actions': next_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q - self._alpha * next_log_pis

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        Q_target = td_target(
            reward=self._reward_scale * self._placeholders['rewards'],
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return tf.stop_gradient(Q_target)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = self._get_Q_target()

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_observations = flatten_input_structure(
            {**Q_observations, 'env_latents': self.latents})
        Q_inputs = flatten_input_structure({
            'observations': Q_observations, 'actions': self._placeholders['actions']})
        Q_values = self._Q_values = tuple(Q(Q_inputs) for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.compat.v1.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        self._Q_training_ops = Q_training_ops = tuple(
            Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        encoder_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._Q_lr,
            name='Q_encoder_optimizer')

        if self._clip_grad:
            gradients, variables = zip(*encoder_optimizer.compute_gradients(
                tf.reduce_sum(Q_losses),
                var_list=self._encoder_net.trainable_variables))
            gradients = [tf.clip_by_value(grad, -self._clip_value, self._clip_value) for grad in gradients]
            self.Q_enc_training_op = encoder_training_op = encoder_optimizer.apply_gradients(zip(gradients, variables))
        else:
            self._Q_encoder_training_op = encoder_training_op = encoder_optimizer.minimize(
               loss=tf.reduce_sum(Q_losses),
               var_list=self._encoder_net.trainable_variables)

        self._training_ops.update({'Q': tf.group(Q_training_ops)})
        self._training_ops.update({'Q_encoder': encoder_training_op})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """

        observations = {
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        }
        policy_inputs = flatten_input_structure(
            {**observations, 'env_latents': self.latents})
        actions = self._policy.actions(policy_inputs)
        log_pis = self._policy.log_pis(policy_inputs, actions)

        log_alpha = self._log_alpha = tf.compat.v1.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_observations = flatten_input_structure(
            {**Q_observations, 'env_latents': self.latents})
        Q_inputs = flatten_input_structure({
            'observations': Q_observations, 'actions': actions})
        Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        self._policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        if self._clip_grad:
            gradients, variables = zip(*self._policy_optimizer.compute_gradients(
                policy_loss,
                var_list=self._policy.trainable_variables))
            gradients = [tf.clip_by_value(grad, -self._clip_value, self._clip_value) for grad in gradients]
            self._policy_train_op = policy_train_op = self._policy_optimizer.apply_gradients(zip(gradients, variables))
        else:
            self._policy_train_op = policy_train_op = self._policy_optimizer.minimize(
                loss=policy_loss,
                var_list=self._policy.trainable_variables)

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_encoder_update(self):
        observations = {
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        }
        next_observations = {
            'next_{}'.format(name): self._placeholders['next_observations'][name]
            for name in self._policy.observation_keys
        }
        ts = tf.reshape(tf.cast(self._placeholders['episodes'], tf.int64), [-1])

        ###### CONTEXT ENCODER #######
        encoder_inputs = flatten_input_structure({
            **observations,
            **next_observations,
            'actions': self._placeholders['actions'],
            'rewards': self._placeholders['rewards'],
        })
        encoder_inputs = tf.concat(encoder_inputs, axis=-1)

        with tf.variable_scope('context_encoder'):
            if self._stochastic_encoder:
                self._encoder_net = feedforward_model(
                    hidden_layer_sizes=self._encoder_size,
                    output_size=2 * self._latent_dim)
                params = self._encoder_net(encoder_inputs)
                self.mu = mu = params[:, :self._latent_dim]
                log_sigma_sq = params[:, self._latent_dim:]
                eps = tf.random.normal(shape=tf.shape(mu))
                self.latents = self.next_latents = latents = mu + eps * tf.sqrt(tf.exp(log_sigma_sq))
            else:
                self._encoder_net = feedforward_model(
                    hidden_layer_sizes=self._encoder_size,
                    output_size=self._latent_dim)
                mu = self._encoder_net(encoder_inputs)
                self.latents = self.next_latents = latents = mu
        ##############################

        if self._batch_tasks:
            task_latents = tf.reshape(latents, [self._task_batch_size, self._batch_size, self._latent_dim])
            latents = tf.expand_dims(tf.reduce_mean(task_latents, axis=1), axis=1)
            latents = tf.reshape(tf.tile(latents, [1, self._batch_size, 1]), [-1, self._latent_dim])
            self.latents = self.next_latents = latents

        ###### PRIOR OVER CONTEXT ######
        with tf.variable_scope('latent_prior'):
            self.lstm_cell = lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=self._latent_dim)
            state = zero_state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)

            seq_len = 200
            batch_size = self._max_episodes // seq_len
            state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            
            cached_latents = tf.reshape(self._placeholders['cached_latents'], 
                [self._max_episodes // seq_len, seq_len, self._latent_dim])

            latent_priors = []
            for i in range(1,seq_len+1):
                curr_latent, state = lstm_cell(cached_latents[:, i-1], state)
                latent_priors.append(curr_latent)

            latent_priors = tf.stack(latent_priors, axis=1)
            latent_priors = tf.reshape(latent_priors,
                [self._max_episodes, self._latent_dim])
            z0 = tf.cast(np.zeros(self._latent_dim), tf.float32)
            self._latent_priors = tf.concat(
                [[z0], latent_priors[:-1]], axis=0,
                name='latent_priors')

            latent_prior = tf.gather(self._latent_priors, ts)
        ##############################

        if self._stochastic_encoder:
            sigma_sq = tf.exp(log_sigma_sq)
            sigma = tf.sqrt(sigma_sq)
            self._encoder_kl_losses = encoder_kl_losses = tf.log(self._prior_sigma / sigma) + \
                (sigma_sq + tf.square(mu - latent_prior)) / (2 * self._prior_sigma_sq) - 0.5
        else:
            self._encoder_kl_losses = encoder_kl_losses = tf.reduce_sum(tf.square(mu - latent_prior), axis=-1)
        encoder_kl_loss = tf.reduce_mean(encoder_kl_losses)

        ###### CONTEXT DECODER ######
        if self._recon_loss:
            decoder_inputs = flatten_input_structure({
                **observations,
                'actions': self._placeholders['actions'],
                'latents': self.latents
            })

            decoder_inputs = tf.concat(decoder_inputs, axis=-1)

            with tf.variable_scope('context_decoder'):
                self._decoder_net = feedforward_model(
                    hidden_layer_sizes=self._decoder_size,
                    output_size=self._state_size+1)
                out = self._decoder_net(decoder_inputs)
                r_delta_obs = out[:, :-1]
                r_rew = out[:, -1:]

            delta_obs_labels = self._placeholders['next_observations']['observations'] - \
                self._placeholders['observations']['observations']
            reward_labels = self._placeholders['rewards']
            self._decoder_losses = tf.square(r_rew - reward_labels) #+ tf.reduce_sum(tf.square(r_delta_obs - delta_obs_labels), axis=-1)

            decoder_loss = tf.reduce_mean(self._decoder_losses)

            self._decoder_train_op = decoder_train_op = self._decoder_optimizer.minimize(
                decoder_loss,
                var_list=self._decoder_net.trainable_variables + self._encoder_net.trainable_variables)

            self._training_ops.update({'decoder_train_op': decoder_train_op})
        ##############################

        self._encoder_train_op = encoder_train_op = self._encoder_optimizer.minimize(
            self._placeholders['beta'] * encoder_kl_loss,
            var_list=self._encoder_net.trainable_variables)

        self._prior_train_op = prior_train_op = self._prior_optimizer.minimize(
            encoder_kl_loss,
            var_list=lstm_cell.trainable_variables)

        self._training_ops.update({'encoder_train_op': encoder_train_op})
        self._training_ops.update({'prior_train_op': prior_train_op})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
            ('Q_loss', self._Q_losses),
            ('policy_loss', self._policy_losses),
            ('alpha', self._alpha),
            ('encoder_kl_loss', self._encoder_kl_losses),
            ('beta', self._placeholders['beta']),
        ))

        if self._recon_loss:
            diagnosables.update(OrderedDict([
                ('decoder_loss', self._decoder_losses)
            ]))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))

        self._diagnostics_ops = OrderedDict([
            (f'{key}-{metric_name}', metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _update_cache(self, encs, ts):
        ts = np.squeeze(ts)
        min_t, max_t = np.min(ts), np.max(ts)
        for t in range(min_t, max_t+1):
            if t == 0:
                continue
            new_entries = encs[np.where(ts == t)]
            if new_entries.shape[0] == 0:
                continue
            m = len(new_entries)
            if m > 0:
                n = self._cache_counts[t]
                if n == 0:
                    self._cached_latents[t] = np.mean(new_entries, 0)
                else:
                    new_average = self._cached_latents[t] * (n - m) / n + np.sum(new_entries, 0) / n
                    self._cached_latents[t] = new_average
                self._cache_counts[t] += m

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        if self._stochastic_encoder:
            enc = self._session.run(self.mu, feed_dict)
        else:
            enc = self._session.run(self.latents, feed_dict)
        self._update_cache(enc, batch['episodes'])

        feed_dict = self._get_feed_dict(iteration, batch)

        if iteration < self._pretrain_iters:
            if self._recon_loss:
                self._session.run([
                    self._encoder_train_op,
                    self._prior_train_op,
                    self._decoder_train_op,
                ], feed_dict)
            else:
                self._session.run([
                    self._encoder_train_op,
                    self._prior_train_op,
                ], feed_dict)
        else:
            self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }

        if iteration is not None:
            feed_dict[self._placeholders['iteration']] = iteration
        feed_dict[self._placeholders['cached_latents']] = self._cached_latents

        if self._anneal_beta:
            self._beta = 0 if iteration <= self._pretrain_iters else min(1e-6 * (iteration - self._pretrain_iters), 1)
        feed_dict[self._placeholders['beta']] = self._beta

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        # TODO(hartikainen): We need to unwrap self._diagnostics_ops from its
        # tensorflow `_DictWrapper`.
        diagnostics = self._session.run({**self._diagnostics_ops}, feed_dict)

        observations = {
            name: batch['observations'][name]
            for name in self._policy.observation_keys
        }
        ts = np.squeeze(batch['episodes']).astype(np.int64)
        latents = self._session.run(self.latents, feed_dict)
        inputs = flatten_input_structure({
            **observations,
            'env_latents': latents
        })

        diagnostics.update(OrderedDict([
            (f'policy/{key}', value)
            for key, value in
            self._policy.get_diagnostics(inputs).items()
        ]))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
