from wrappers import ResizeWrapper, NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, CropImageWrapper
from gym_duckietown.simulator import Simulator
from ray.tune.registry import register_env


def reg_env(args):

    seed = args.seed
    map_name = args.map_name
    max_steps = args.max_steps
    domain_rand = False
    camera_width = 640
    camera_height = 480
    
    def prepare_env(env_config):
        env = Simulator(
            seed=seed,
            map_name=map_name,
            max_steps=max_steps,
            domain_rand=domain_rand,
            camera_width=camera_width,
            camera_height=camera_height,
        )

        env = CropImageWrapper(env,3)
        env = ResizeWrapper(env)
        env = NormalizeWrapper(env)
        env = ActionWrapper(env)  
        env = DtRewardWrapper(env)

        return env


    register_env("TTenv", prepare_env)

def get_env(args):

    seed = args.seed
    map_name = args.map_name
    max_steps = args.max_steps
    domain_rand = False
    camera_width = 640
    camera_height = 480
    
    env = Simulator(
        seed=seed,
        map_name=map_name,
        max_steps=max_steps,
        domain_rand=domain_rand,
        camera_width=camera_width,
        camera_height=camera_height,
        )

    env = CropImageWrapper(env,3)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ActionWrapper(env)  
    env = DtRewardWrapper(env)

    return env