import ray
import logging
import agents
import argparse
import utils
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer
import time

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, type=str, required=True, help="Checkpoint path")
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--mode", default='impala', type=str)
parser.add_argument("--map_name", default='TensorTesok_custom.yaml', type=str)
parser.add_argument("--max_steps", default=100000, type=int)
args = parser.parse_args()

utils.reg_env(args)

trainer_config = {
	"env": "TTenv",         
	"framework": "torch",

	"model": {
		"fcnet_activation": "relu"
	},
	"env_config": {
			"accepted_start_angle_deg": 5,
		},
}

ray.shutdown()

ray.init(
    num_cpus=3,
    num_gpus=0,
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,
)

if args.mode == "impala":
    model = ImpalaTrainer(config=trainer_config, env="TTenv")
elif args.mode == "ppo": 
    model = PPOTrainer(config=trainer_config, env="TTenv")

model.restore(args.checkpoint)
env = utils.get_env(args)

obs = env.reset()
for step in range(10):
    obs = env.reset()
    env.render()
    done = False
    while not done:
        action = model.compute_action(obs)
        observation, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.02)
env.close()
