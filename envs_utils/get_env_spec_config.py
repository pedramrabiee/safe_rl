def get_env_spec_config(train_env):
    env_config = None
    if train_env['env_collection'] == 'gym':
        if train_env['env_id'] == 'Pendulum-v0':
            from envs_utils.gym.pendulum.pendulum_configs import env_config
        else:
            raise NotImplementedError
    elif train_env['env_collection'] == 'safety_gym':
        if train_env['env_id'] == 'Point':
            from envs_utils.safety_gym.point_robot_configs import env_config
            from envs_utils.safety_gym.point_robot_configs import engine_config
            env_config.update(engine_config)
    else:
        raise NotImplementedError
    return env_config
