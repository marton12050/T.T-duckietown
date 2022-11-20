
from gym_duckietown.simulator import Simulator
from gym_duckietown.wrappers import *

import logging
import ray
from ray.tune.registry import register_env

from wrappers import ResizeWrapper, NormalizeWrapper, DtRewardWrapper, ActionWrapper
import agents
import argparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--map_name", default='small_loop', type=str)
parser.add_argument("--max_steps", default=100000, type=int)
parser.add_argument("--training_name", default='Training_results', type=str)
parser.add_argument("--num_cpus", default=1, type=int)
parser.add_argument("--num_gpus", default=0, type=int)
parser.add_argument("--mode", default='ppo', type=str)
args = parser.parse_args()

seed = args.seed
map_name = args.map_name
max_steps = args.max_steps
domain_rand = False
camera_width = 640
camera_height = 480
checkpoint_path = "./dump"
training_name = args.training_name
mode = args.mode


def prepare_env(env_config):
    env = Simulator(
        seed=seed,
        map_name=map_name,
        max_steps=max_steps,
        domain_rand=domain_rand,
        camera_width=camera_width,
        camera_height=camera_height,
    )

    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    return env


register_env("TTenv", prepare_env)

ray.shutdown()

ray.init(
    num_cpus=args.num_cpus,
    num_gpus=args.num_gpus,
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,
)


if mode == 'ppo':
    analysis = agents.ppo(training_name, args.num_gpus, args.num_cpus-1, max_steps)
