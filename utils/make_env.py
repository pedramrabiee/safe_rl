import gym
from utils.wrappers.monitor import Monitor
from utils.wrappers.scaler import ActionScalerWrapper


def make_env(env_id,
             collection='gym',
             ac_lim=(-1.0, 1.0),
             video_dict=None,
             max_episode_time=None,
             use_custom_env=False):

    env = None
    max_episode_len = None
    if collection == 'gym':
        env = gym.make(env_id)
        env = customize_env(env_id, env) if use_custom_env else env
        if max_episode_time:
            env._max_episode_steps = int(max_episode_time / env.unwrapped.dt)   # _max_episode_steps is an attribute of TimeLimit
        max_episode_len = env._max_episode_steps

    elif collection == 'safety_gym':
        # For the safety_gym collection, all the environment customization (e.g., timestep, max_episode_steps, etc.)
        # are done using config dictionary and MuJoCo model xml file
        from safety_gym.envs.engine import Engine
        env_config = None
        if env_id == 'Point':
            from configs.env_configs.safety_gym_envs.point_robot_configs import env_config
        env = Engine(env_config)
        if max_episode_time:
            env.num_steps = int(max_episode_time / env.robot.sim.model.opt.timestep)
        max_episode_len = env.num_steps

    env = ActionScalerWrapper(env, ac_lim=ac_lim)

    if video_dict is not None:
        env = Monitor(env,
                      video_dict['video_save_dir'],
                      video_callable=video_dict['video_callable'],
                      force=True,
                      write_upon_reset=False,
                      )
    env_info = dict(max_episode_len=max_episode_len)
    return env, env_info


def customize_env(env_id, env):
    if env_id == 'Pendulum-v0':
        from configs.env_configs.gym_envs.inverted_pendulum_configs import inverted_pendulum_customize
        return inverted_pendulum_customize(env)
    else:
        raise NotImplementedError