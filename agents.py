from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer
import ray


def impala(name, num_gpus, num_workers, max_steps, restore_chk):
        trainer_history = ray.tune.run(
                ImpalaTrainer,#Set agent algorithm
                name=name,
                config={
                        "env": "TTenv",
                        "framework": "torch",
                        #Env specific config
                        "model": {
                                "fcnet_activation": "relu"
                        },

                        "env_config": {
                                "accepted_start_angle_deg": 5,
                        },

                        #Agent specific config
                        "num_workers": num_workers,
                        "num_gpus": num_gpus,
                        "train_batch_size": ray.tune.choice([1500, 2000]),
                        "decay": ray.tune.loguniform(0.99, 0.995),
                        "grad_clip":40,
                        "epsilon": ray.tune.loguniform(0.1, 0.14),
                        "gamma": 0.99,  
                        # Always die after a few hundrew step,smaller learning rate a little bit longer
                        # (probably env or ray problem from what I saw in the github issues)
                        "lr": ray.tune.loguniform(0.0001, 5e-6),
                        "sgd_minibatch_size": ray.tune.choice([150, 200]),
                        "num_sgd_iter": ray.tune.choice([2, 4, 8, 16]),
                        "lambda": 0.95,
                },
                stop={'timesteps_total': max_steps},
                restore=restore_chk,
                checkpoint_at_end=True,
                local_dir="./temp",
                keep_checkpoints_num=2,
                checkpoint_score_attr="episode_reward_mean",
                checkpoint_freq=1,
                metric="episode_reward_mean",
                mode="max",
        )

        return trainer_history


def ppo(name, num_gpus, num_workers, max_steps, restore_chk):
        trainer_history = ray.tune.run(
                PPOTrainer,#Set agent algorithm
                name=name,
                config={
                        "env": "TTenv",
                        "framework": "torch",

                        #Env specific config
                        "model": {
                                "fcnet_activation": "relu"
                        },
                        "env_config": {
                                "accepted_start_angle_deg": 5,
                        },

                        #Agent specific config
                        "num_workers": num_workers,
                        "num_gpus": num_gpus,
                        "train_batch_size": ray.tune.choice([1500, 2000]), #at 4096 my laptop crashing
                        "lr": ray.tune.loguniform(0.0001, 5e-6),
                        "clip_param": ray.tune.loguniform(0.28, 0.32),
                        "gamma": 0.99,  
                        "sgd_minibatch_size": ray.tune.choice([150, 200]),
                        #"num_sgd_iter": random.choice([1, 2, 4, 8, 16]),
                        "lambda": 0.98,
                        "kl_target": ray.tune.loguniform(0.01, 0.015),
                        "rollout_fragment_length": 200,
                },
                stop={'timesteps_total': max_steps},
                restore=restore_chk,
                checkpoint_at_end=True,
                local_dir="./temp",
                keep_checkpoints_num=2,
                checkpoint_score_attr="episode_reward_mean",
                checkpoint_freq=1,
                metric="episode_reward_mean",
                mode="max",
        )

        return trainer_history