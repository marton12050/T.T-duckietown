from ray.rllib.agents.ppo import PPOTrainer
from ray.air.callbacks.wandb import WandbLoggerCallback

import ray


def ppo(name, max_steps):
    parameter_search_analysis = ray.tune.run(
        PPOTrainer,
        name=name,
        config={
            "env": "TTenv",
            "framework": "torch",

            "model": {
                "fcnet_activation": "relu"
            },

            "num_workers": 3,
            "train_batch_size": ray.tune.choice([1024, 2048]),
            "gamma": 0.90,
            "lr": ray.tune.loguniform(0.0001, 5e-6),
            "sgd_minibatch_size": ray.tune.choice([128, 512]),
            "lambda": 0.95,
        },

        stop={'timesteps_total': max_steps},
        metric="episode_reward_mean",
        mode="max",
        checkpoint_at_end=True,
        local_dir="./temp",
        keep_checkpoints_num=2,
        checkpoint_score_attr="episode_reward_mean",
        checkpoint_freq=1,
        callbacks=[WandbLoggerCallback(
            project="T.T-duckietown",
            api_key_file="wanadb_api_key",
            log_config=True)]
    )

    return parameter_search_analysis
