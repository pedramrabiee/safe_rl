def get_env_spec_config(train_env):
    config = None
    if train_env['env_collection'] == 'gym':
        if train_env['env_id'] == 'Pendulum-v0':
            from configs.env_configs.gym_envs.inverted_pendulum_configs import config
        else:
            raise NotImplementedError
    elif train_env['env_collection'] == 'safety_gym':
        if train_env['env_id'] == 'Point':
            from configs.env_configs.safety_gym_envs.point_robot_configs import config
    else:
        raise NotImplementedError
    return config
