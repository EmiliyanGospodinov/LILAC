import numpy as np
from gym.spaces import Dict

from .simple_replay_pool import SimpleReplayPool


class MultitaskReplayPool(object):
    def __init__(self,
                 environment,
                 total_tasks=7500,
                 *args,
                 extra_fields=None,
                 **kwargs):
        observation_space = environment.observation_space
        action_space = environment.action_space
        assert isinstance(observation_space, Dict), observation_space
    
        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space
        self._total_tasks = total_tasks
        self._current_task = 0
        self._task_pools = dict([(idx, SimpleReplayPool(
            environment=environment,
            *args,
            **kwargs
        )) for idx in range(self._total_tasks)])

    def add_sample(self, sample):
        self._task_pools[self._current_task].add_sample(sample)

    def terminate_episode(self):
        self._task_pools[self._current_task].terminate_episode()
        self._current_task += 1

    @property
    def size(self):
        total_size = 0
        for idx in range(self._total_tasks):
            total_size += self._task_pools[idx].size
        return total_size

    def add_path(self, path):
        self._task_pools[self._current_task].add_path(path)

    def random_task_indices(self, task_batch_size):
        return np.random.randint(0, self._current_task, task_batch_size)

    def random_batch(self, batch_size):
        batch = {
            'observations': {'observations': []},
            'next_observations': {'observations': []},
            # 'observations': {'observation': []},
            # 'next_observations': {'observation': []},
            'actions': [],
            'rewards': [],
            'terminals': [],
            'episodes': [],
        }

        tasks = self.random_task_indices(batch_size//8)
        for task in tasks:
            task_batch = self._task_pools[task].random_batch(8)
            for key, value in task_batch.items():
                if key in ['observations', 'next_observations']:
                    batch[key]['observations'].append(task_batch[key]['observations'])
                    # batch[key]['observation'].append(task_batch[key]['observation'])
                elif key in ['actions', 'rewards', 'terminals', 'episodes']:
                    batch[key].append(task_batch[key])

        for key in batch:
            if key in ['observations', 'next_observations']:
                batch[key]['observations'] = np.concatenate(batch[key]['observations'], 0)
                # batch[key]['observation'] = np.concatenate(batch[key]['observation'], 0)
            else:
                batch[key] = np.concatenate(batch[key], 0)
        return batch

    def batch_from_index(self, task_index, batch_size=50):
        if task_index >= self._current_task:
            return None
        return self._task_pools[task_index].random_batch(batch_size)
