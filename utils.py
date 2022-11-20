def get_env(args):
    seed = args.seed
    map_name = args.map_name
    max_steps = args.max_steps
    domain_rand = False
    camera_width = 640
    camera_height = 480

    def prepare_env(env_config):
        e = Simulator(
            seed=seed,
            map_name=map_name,
            max_steps=max_steps,
            domain_rand=domain_rand,
            camera_width=camera_width,
            camera_height=camera_height,
        )

        e = ResizeWrapper(e)
        e = NormalizeWrapper(e)
        e = ActionWrapper(e)
        e = DtRewardWrapper(e)

        return e

    register_env("TTenv", prepare_env)
    return prepare_env("")
