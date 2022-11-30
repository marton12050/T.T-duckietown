import ray
import logging
import agents
import argparse
import utils

#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--map_name", default='udem1', type=str)
parser.add_argument("--max_steps", default=1000000, type=int)
parser.add_argument("--training_name", default='Training_results', type=str)
parser.add_argument("--num_cpus", default=3, type=int)
parser.add_argument("--num_gpus", default=0, type=int)
parser.add_argument("--mode", default='impala', type=str)
parser.add_argument("--checkpoint", default=None, type=str)
args = parser.parse_args()

max_steps = args.max_steps
checkpoint_path = "./temp"
training_name = args.training_name
mode = args.mode
restore_ckp = args.checkpoint

utils.reg_env(args)

ray.shutdown()

ray.init(
    num_cpus=args.num_cpus,
    num_gpus=args.num_gpus,
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,
)
if mode == 'impala':
    history = agents.impala(training_name, args.num_gpus, args.num_cpus-1, max_steps, restore_ckp)
elif mode == 'ppo':
    history = agents.ppo(training_name, args.num_gpus, args.num_cpus-1, max_steps, restore_ckp)


print(f"""
###############TRAINING FINISHED#############
Best checkpoint path:{history.best_checkpoint}
""")

