import ray
import agents
import argparse
import utils

#Read given parameters
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--map_name", default='small_loop', type=str)
parser.add_argument("--max_steps", default=2000000, type=int)
parser.add_argument("--training_name", default='Trained_models', type=str)
parser.add_argument("--num_cpus", default=3, type=int)
parser.add_argument("--num_gpus", default=0, type=int)
parser.add_argument("--mode", default='ppo', type=str)
parser.add_argument("--checkpoint", default=None, type=str)
args = parser.parse_args()


# Register the duckietown env to a ray register for use (utils.py has more detail)
utils.reg_env(args) 

#Start ray instance, if running restart it
ray.shutdown()
ray.init(
    num_cpus=args.num_cpus,
    num_gpus=args.num_gpus,
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,
)

#Select agent to train with
if args.mode == 'impala':
    history = agents.impala(args.training_name, args.num_gpus, args.num_cpus-1, args.max_steps, args.checkpoint)
elif args.mode == 'ppo':
    history = agents.ppo(args.training_name, args.num_gpus, args.num_cpus-1, args.max_steps, args.checkpoint)

#Training ended print out the best checkpoint path
print(f"""
###############TRAINING FINISHED#############
Best checkpoint path:{history.best_checkpoint}
""")

