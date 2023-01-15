import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class HalfCheetahWindVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_wind.xml',
                 obs_dp=False,
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0.5,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self._wind_frc = 0.
        self._target_vel = 0.
        self._meta_time = -1
        self._avg_f = 0.
        self._mag_f = 10.
        self._dtheta_f = 0.2
        self._avg_v = 1.5
        self._mag_v = 1.5
        self._dtheta_v = 0.5
        self._obs_dp = obs_dp

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        self.action_space.low[-1] = 0.0
        self.action_space.high[-1] = 0.0

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        action[-1] = self._wind_frc
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        # x_velocity = ((x_position_after - x_position_before)
        #               / self.dt)
        x_velocity = self.sim.data.qvel[0]

        ctrl_cost = self.control_cost(action[:-1])

        forward_reward = -1.0 * abs(x_velocity - self._target_vel)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._obs_dp:
            observation = np.concatenate((position, velocity, [0.01*self._wind_frc,self._target_vel])).ravel()
        else:
            observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        self._wind_frc = self._avg_f + self._mag_f * np.sin(self._dtheta_f * self._meta_time)
        self._target_vel = self._avg_v + self._mag_v * np.sin(self._dtheta_v * self._meta_time)
        self._meta_time += 1

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
