from gym_duckietown.simulator import Simulator
from gym_duckietown.wrappers import *

import logging
import ray
from ray.tune.registry import register_env

from wrappers import ResizeWrapper, NormalizeWrapper, DtRewardWrapper, ActionWrapper
import utils
import agents
import argparse
import time


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--mode", default='ppo', type=str)
parser.add_argument("--map_name", default='udem1', type=str)
parser.add_argument("--max_steps", default=100000, type=int)
parser.add_argument("--training_name", default='PPO_training_ckp', type=str)

args = parser.parse_args()

utils.get_env(args)

checkpoint_path = "./temp"
training_name = args.training_name + f"_{int(time.time())}"
mode = args.mode


ray.shutdown()

ray.init(
    num_cpus=args.num_cpus,
    num_gpus=args.num_gpus,
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,
)


if mode == 'ppo':
    analysis = agents.ppo(training_name, max_steps)
