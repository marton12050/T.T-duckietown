from wrappers import ResizeWrapper, NormalizeWrapper, DtRewardWrapper, ActionWrapper, CropImageWrapper, DiscreteWrapper
from gym_duckietown.simulator import Simulator
from ray.tune.registry import register_env

# Register the duckietown env to a ray register for use
def reg_env(args):

    #Set parameters about the duckietown env, like camera image size
    domain_rand = False
    camera_width = 640
    camera_height = 480
    
    #Function to register the env as specified in ray
    def duckie_env(env_config):
        #Create a duckietown simulator based from original repository
        env = Simulator(
            seed=args.seed,
            map_name=args.map_name,
            max_steps=args.max_steps,
            domain_rand=domain_rand,
            camera_width=camera_width,
            camera_height=camera_height,
        )
        env = CropImageWrapper(env,3)
        env = ResizeWrapper(env)
        env = NormalizeWrapper(env)
        env = DiscreteWrapper(env)  
        #env = ActionWrapper(env)  
        env = DtRewardWrapper(env)

        return env #Return the created simulator with wrappers around it

    register_env("TTenv", duckie_env)# Register env for Ray

#Same as reg_env, but for testing to get an env instance
def get_env(args):
    domain_rand = False
    camera_width = 640
    camera_height = 480
    
    env = Simulator(
        seed=args.seed,
        map_name=args.map_name,
        max_steps=args.max_steps,
        domain_rand=domain_rand,
        camera_width=camera_width,
        camera_height=camera_height,
        )

    env = CropImageWrapper(env,3)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = DiscreteWrapper(env)  
    #env = ActionWrapper(env)  
    env = DtRewardWrapper(env)

    return env