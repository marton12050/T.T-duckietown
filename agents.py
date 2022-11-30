from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer
import ray


def impala(name, num_gpus, num_workers, max_steps, restore_chk):
        trainer_history = ray.tune.run(
                ImpalaTrainer,
                name=name,
                config={
                        "env": "TTenv",
                        "framework": "torch",

                        "model": {
                                "fcnet_activation": "relu"
                        },

                        "env_config": {
                                "accepted_start_angle_deg": 5,
                        },

                        "num_workers": num_workers,
                        "num_gpus": num_gpus,
                        "train_batch_size": 1024, #Bigger batch size and my laptop crash lmao
                        "gamma": 0.99,
                        # If learning rate is too big, it will crash after around 100K steps(lr=0.0001)
                        "lr": ray.tune.loguniform(0.00005, 5e-6),
                        "seed":123,
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
                PPOTrainer,
                name=name,
                config={
                        "env": "TTenv",
                        "framework": "torch",

                        "model": {
                                "fcnet_activation": "relu"
                        },

                        "env_config": {
                                "accepted_start_angle_deg": 5,
                        },
                        "num_workers": num_workers,
                        "num_gpus": num_gpus,
                        "train_batch_size": 1024,
                        "gamma": 0.99,  
                        "lr": ray.tune.loguniform(0.0001, 5e-6),
                        "sgd_minibatch_size": 128,
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