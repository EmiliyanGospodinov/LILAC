"""Custom Gym environments.

Every class inside this module should extend a gym.Env class. The file
structure should be similar to gym.envs file structure, e.g. if you're
implementing a mujoco env, you would implement it under gym.mujoco submodule.
"""

import gym


CUSTOM_GYM_ENVIRONMENTS_PATH = __package__
MUJOCO_ENVIRONMENTS_PATH = f'{CUSTOM_GYM_ENVIRONMENTS_PATH}.mujoco'

MUJOCO_ENVIRONMENT_SPECS = (
    {
        'id': 'Swimmer-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.swimmer_v3:SwimmerEnv'),
    },
    {
        'id': 'Hopper-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.hopper_v3:HopperEnv'),
    },
    {
        'id': 'Walker2d-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.walker2d_v3:Walker2dEnv'),
    },
    {
        'id': 'HalfCheetah-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv'),
    },
    {
        'id': 'Ant-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.ant_v3:AntEnv'),
    },
    {
        'id': 'Humanoid-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.humanoid_v3:HumanoidEnv'),
    },
    {
        'id': 'Pusher2d-Default-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:Pusher2dEnv'),
    },
    {
        'id': 'Pusher2d-DefaultReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:ForkReacherEnv'),
    },
    {
        'id': 'Pusher2d-ImageDefault-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImagePusher2dEnv'),
    },
    {
        'id': 'Pusher2d-ImageReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImageForkReacher2dEnv'),
    },
    {
        'id': 'Pusher2d-BlindReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:BlindForkReacher2dEnv'),
    },
)

GENERAL_ENVIRONMENT_SPECS = (
    {
        'id': 'MultiGoal-Default-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.multi_goal:MultiGoalEnv')
    },
)

MULTIWORLD_ENVIRONMENT_SPECS = (
    {
        'id': 'Point2DEnv-Default-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DEnv'
    },
    {
        'id': 'Point2DEnv-Wall-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DWallEnv'
    },
)

METAWORLD_ENVIRONMENT_SPECS = (
    {
        'id': 'SawyerReachPushPickPlaceEnv-Default-v0',
        'entry_point': 'metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place:SawyerReachPushPickPlaceEnv'
    },
)

PYBULLET_ENVIRONMENT_SPECS = (
    {
        'id': 'MinitaurGoalVelEnv-Default-v0',
        'entry_point': 'gym.envs.mujoco.minitaur_goal_vel_env:MinitaurGoalVelocityEnv'
    },
)

MUJOCO_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MUJOCO_ENVIRONMENT_SPECS)


GENERAL_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in GENERAL_ENVIRONMENT_SPECS)


MULTIWORLD_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MULTIWORLD_ENVIRONMENT_SPECS)


METAWORLD_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in METAWORLD_ENVIRONMENT_SPECS)


PYBULLET_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in PYBULLET_ENVIRONMENT_SPECS)


GYM_ENVIRONMENTS = (
    *MUJOCO_ENVIRONMENTS,
    *GENERAL_ENVIRONMENTS,
    *MULTIWORLD_ENVIRONMENTS,
    *METAWORLD_ENVIRONMENTS,
    *PYBULLET_ENVIRONMENTS,
)


def register_mujoco_environments():
    """Register softlearning mujoco environments."""
    for mujoco_environment in MUJOCO_ENVIRONMENT_SPECS:
        gym.register(**mujoco_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MUJOCO_ENVIRONMENT_SPECS)

    return gym_ids


def register_general_environments():
    """Register gym environments that don't fall under a specific category."""
    for general_environment in GENERAL_ENVIRONMENT_SPECS:
        gym.register(**general_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  GENERAL_ENVIRONMENT_SPECS)

    return gym_ids


def register_multiworld_environments():
    """Register custom environments from multiworld package."""
    for multiworld_environment in MULTIWORLD_ENVIRONMENT_SPECS:
        gym.register(**multiworld_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MULTIWORLD_ENVIRONMENT_SPECS)

    return gym_ids


def register_metaworld_environments():
    """Register custom environments from metaworld package."""
    for metaworld_environment in METAWORLD_ENVIRONMENT_SPECS:
        gym.register(**metaworld_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  METAWORLD_ENVIRONMENT_SPECS)

    return gym_ids


def register_pybullet_environments():
    """Register custom environments from pybullet package."""
    for pybullet_environment in PYBULLET_ENVIRONMENT_SPECS:
        gym.register(**pybullet_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  PYBULLET_ENVIRONMENT_SPECS)

    return gym_ids


def register_environments():
    registered_mujoco_environments = register_mujoco_environments()
    registered_general_environments = register_general_environments()
    registered_multiworld_environments = register_multiworld_environments()
    registered_metaworld_environments = register_metaworld_environments()
    registered_pybullet_environments = register_pybullet_environments()

    return (
        *registered_mujoco_environments,
        *registered_general_environments,
        *registered_multiworld_environments,
        *registered_metaworld_environments,
        *registered_pybullet_environments,
    )
