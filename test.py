from gym_duckietown.simulator import Simulator
from gym_duckietown.wrappers import *

import logging
import ray
from ray.tune.registry import register_env

from wrappers import ResizeWrapper, NormalizeWrapper, DtRewardWrapper, ActionWrapper
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, type=str)
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--mode", default='ppo', type=str)
parser.add_argument("--map_name", default='TensorTesok_custom.yaml', type=str)
parser.add_argument("--max_steps", default=100000, type=int)
args = parser.parse_args()

env = utils.get_env(args)

trainer_config = {
    "env": "TTenv",
    "framework": "torch",

    "model": {
        "fcnet_activation": "relu"
    }
}

model = PPOTrainer(config=trainer_config, env="TTenv")
model.restore(args.checkpoint)


obs = env.reset()
for step in range(20):
    action, _ = model.compute_single_action(obs)
    print('Action made: ', action)
    observation, reward, done, info = env.step(action)
    print('After the action: ', observation, reward, done, info)
    env.render()
    time.sleep(0.5)
env.close()
