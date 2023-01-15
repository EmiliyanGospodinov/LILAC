# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom Minitaur environment with target velocity.

Implements minitaur environment with rewards dependent on closeness to goal
velocity. Extends the MinitaurExtendedEnv class from PyBullet
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from pybullet_envs.minitaur.envs import minitaur_extended_env

ENV_DEFAULTS = {
  "accurate_motor_model_enabled": True,
  "never_terminate": False,
  "history_length": 0,
  "urdf_version": "rainbow_dash_v0",
  "history_include_actions": False,
  "control_time_step": 0.02,
  "history_include_states": False,
  "include_leg_model": True
}

class MinitaurGoalVelocityEnv(minitaur_extended_env.MinitaurExtendedEnv):
  """The 'extended' minitaur env with a target velocity."""

  def __init__(self,
               ns_mass=True,
               goal_vel=0.5,
               goal_limit=0.8,
               max_steps=500,
               debug=False,
               obs_dp=False,
               **kwargs):
    self.set_sample_goal_args(goal_limit, goal_vel)
    self._current_vel = 0.
    self._debug = debug
    self._max_steps = max_steps

    self._meta_time = -1
    self._obs_dp = obs_dp

    if not kwargs:
      kwargs = ENV_DEFAULTS
    super(MinitaurGoalVelocityEnv, self).__init__(**kwargs)
    self.set_foot_friction(friction)

    self.dt = self.control_time_step

    self._ns_m = ns_mass   # nonstationary mass

    mass = self.minitaur.GetBaseMassesFromURDF()
    self._avg_m = [m * 1.0 for m in mass]
    self._mag_m = [m * 0.5 for m in mass]
    self._dtheta_m = 0.3

  @property
  def current_vel(self):
    return self._current_vel

  def _termination(self):
    """Determines whether the env is terminated or not.

    Checks whether 1) the front leg is bent too much 2) the time exceeds
    the manually set weights or 3) if the minitaur has "fallen"
    Returns:
      terminal: the terminal flag whether the env is terminated or not
    """
    if self._never_terminate:
      return False

    if self._counter >= self._max_steps:
      return True

    return self.is_fallen()  # terminates automatically when in fallen state

  def is_fallen(self):
    if super(MinitaurGoalVelocityEnv, self).is_fallen():
      return True
    leg_model = self.convert_to_leg_model(self.minitaur.GetMotorAngles())
    swing0 = leg_model[0]
    swing1 = leg_model[2]
    maximum_swing_angle = 0.8
    if swing0 > maximum_swing_angle or swing1 > maximum_swing_angle:
      return True
    return False

  def set_foot_friction(self, friction=None):
    self._foot_friction = friction
    if friction:
      self.minitaur.SetFootFriction(friction)

  def set_sample_goal_args(self, goal_limit=None, goal_vel=None):
    if goal_limit is not None:
      self._goal_limit = goal_limit
    if goal_vel is not None:
      self._goal_vel = goal_vel

  def reset(self, **kwargs):
    if self._ns_m and self._meta_time >= 0:
        self._mass = [a + m * np.sin(self._dtheta_m * self._meta_time) for a, m in zip(self._avg_m, self._mag_m)]
        self.minitaur.SetBaseMasses(self._mass)

    self._meta_time += 1

    if kwargs.get('initial_motor_angles', None):
      return super(minitaur_extended_env.MinitaurExtendedEnv, self).reset(**kwargs)
    return super(MinitaurGoalVelocityEnv, self).reset()

  def reward(self):
    """Compute rewards for the given time step.

    It considers two terms: 1) forward velocity reward and 2) action
    acceleration penalty.
    Returns:
      reward: the computed reward.
    """
    current_base_position = self.minitaur.GetBasePosition()
    dt = self.control_time_step
    self._current_vel = velocity = (current_base_position[0] - self._last_base_position[0]) / dt
    vel_clip = np.clip(velocity, -self._goal_limit, self._goal_limit)
    velocity_reward = self._goal_vel - np.abs(self._goal_vel - vel_clip)

    action = self._past_actions[self._counter - 1]
    prev_action = self._past_actions[max(self._counter - 2, 0)]
    prev_prev_action = self._past_actions[max(self._counter - 3, 0)]
    acc = action - 2 * prev_action + prev_prev_action
    action_acceleration_penalty = np.mean(np.abs(acc))

    reward = 0.0
    reward += 1.0 * velocity_reward
    reward -= 0.01 * action_acceleration_penalty

    if self._debug:
      self.pybullet_client.addUserDebugText('Current velocity: {:3.2f}'.format(
        self._current_vel), [0, 0, 1], [1, 0, 0])
    return reward, velocity_reward, action_acceleration_penalty, velocity

  def step(self, action):
    if self._ns_s:
      action *= self._scale
    next_obs, _, done, info = super(MinitaurGoalVelocityEnv, self).step(action)
    reward, velocity_reward, action_acceleration_penalty, velocity = self.reward()
    info.update(base_reward=reward)
    info.update(velocity_reward=velocity_reward)
    info.update(action_acceleration_penalty=action_acceleration_penalty)
    info.update(velocity=velocity)
    return next_obs, reward, done, info

  def _get_observation(self):
    obs = super(MinitaurGoalVelocityEnv, self)._get_observation()
    if self._obs_dp:
      obs = np.concatenate([obs, [self._goal_vel]])
    return obs
