# LILAC

Our code is built upon the Tensorflow implementation of Soft Actor-Critic at https://github.com/rail-berkeley/softlearning. We experiment with our algorithm in four custom domains:

- Sawyer Reaching: `metaworld/metaworld/envs/mujoco/sawyer_xyz/sawyer_reach_push_pick_place.py`
- Half-Cheetah Vel: `gym/gym/envs/mujoco/half_cheetah_vel_env.py`
- Half-Cheetah WindVel: `gym/gym/envs/mujoco/half_cheetah_windvel.py`
- Minitaur: `gym/gym/envs/mujoco/minitaur_goal_vel_env.py`

# Getting Started

1. Create and activate conda environment
```
cd softlearning
conda env create -f environment.yml
conda activate softlearning_lilac
pip install -e .
```

2. Install gym and metaworld
```
cd ../gym
pip install -e .

cd ../metaworld
pip install -e .

```

3. Install pybullet. Follow build instructions at https://github.com/bulletphysics/bullet3.
```
pip install pybullet
```


4. Run script
```
bash scripts/run_sawyer.sh
```
By default, the data and policy are saved under `~/ray_results/`.