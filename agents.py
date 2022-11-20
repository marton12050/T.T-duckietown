from ray.rllib.agents.ppo import PPOTrainer
from ray.air.callbacks.wandb import WandbLoggerCallback

import ray


def ppo(name, num_gpus, num_workers, max_steps):
        parameter_search_analysis = ray.tune.run(
                PPOTrainer,
                name=name,
                config={
                        "env": "TTenv",
                        "framework": "torch",

                        # Model config
                        "model": {
                                "fcnet_activation": "relu"
                        },

                        "env_config": {
                                "accepted_start_angle_deg": 5,
                        },

                        "num_workers": num_workers,
                        "num_gpus": num_gpus,
                        "train_batch_size": ray.tune.choice([2048, 4096]),
                        "gamma": 0.90,
                        "lr": ray.tune.loguniform(0.0001, 5e-6),
                        "sgd_minibatch_size": ray.tune.choice([128, 256]),
                        "lambda": 0.90,
                },
                stop={'timesteps_total': max_steps},
                metric="episode_reward_mean",
                mode="max",
                checkpoint_at_end=True,
                local_dir="./dump",
                keep_checkpoints_num=2,
                checkpoint_score_attr="episode_reward_mean",
                checkpoint_freq=1,
                callbacks=[WandbLoggerCallback(
                        project="T.T-duckietown",
                        api_key_file="wanadb_api_key",
                        log_config=True)]
        )

        return parameter_search_analysis