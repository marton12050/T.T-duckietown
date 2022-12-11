import ray
import argparse
import utils
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer
import time

#Read given parameters
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, type=str, required=True, help="Checkpoint path")
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--mode", default='ppo', type=str)
parser.add_argument("--map_name", default='TensorTesok_custom.yaml', type=str)
parser.add_argument("--max_steps", default=100000, type=int)
args = parser.parse_args()

# Register the duckietown env to a ray register for use (utils.py has more detail)
utils.reg_env(args)

#Config for the env itself
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

#Start ray instance, if running restart it
ray.shutdown()
ray.init(
    num_cpus=3,
    num_gpus=0,
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,
)

#Select agent to test with
if args.mode == "impala":
    model = ImpalaTrainer(config=trainer_config, env="TTenv")
elif args.mode == "ppo": 
    model = PPOTrainer(config=trainer_config, env="TTenv")

#Restore with the given checkpoint
model.restore(args.checkpoint)
#Create a separate env which the agent cant see
env = utils.get_env(args)

obs = env.reset()
for step in range(10):#Agent has 10 'life' for testing
    obs = env.reset()
    env.render()
    done = False
    while not done:
        action = model.compute_action(obs)#Agent calculate action from observastion
        print(action)
        observation, reward, done, info = env.step(action)#Give action to the env
        env.render()
        time.sleep(0.01)
env.close()
