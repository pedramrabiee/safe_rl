import gym
from utils.wrappers.monitor import Monitor
from utils.wrappers.scaler import ActionScalerWrapper
import importlib


def make_env(env_id,
             env_nickname,
             collection='gym',
             ac_lim=(-1.0, 1.0),
             video_dict=None,
             max_episode_time=None,
             use_custom_env=False,
             make_env_dict=None):

    env = None
    max_episode_len = None
    if collection == 'gym':
        env = gym.make(env_id)
        env = customize_env(env_nickname, env) if use_custom_env else env
        if max_episode_time:
            env._max_episode_steps = int(max_episode_time / env.unwrapped.dt)   # _max_episode_steps is an attribute of TimeLimit
        max_episode_len = env._max_episode_steps

    elif collection == 'safety_gym':
        # For the safety_gym collection, all the environment customization (e.g., timestep, max_episode_steps, etc.)
        # are done using engine_config dictionary and MuJoCo model xml file
        from safety_gym.envs.engine import Engine
        from utils.safety_gym_utils import get_engine_config
        engine_config = get_engine_config(env_id)
        if make_env_dict is not None:
            engine_config.update(make_env_dict)
        env = Engine(engine_config)
        if max_episode_time:
            env.num_steps = int(max_episode_time / env.robot.sim.model.opt.timestep)
        max_episode_len = env.num_steps
    elif collection == 'misc_env':
        parts = env_nickname.split('_')
        class_name = ''.join(part.capitalize() for part in parts)

        class_name = class_name + "Env"
        module_name = f'envs_utils.{collection}.{env_nickname}.{env_nickname}_env'

        try:
            env_module = importlib.import_module(module_name)
            env_cls = getattr(env_module, class_name)
            env = env_cls()
            max_episode_len = env.max_episode_len
        except ImportError:
            # Handle cases where the module or class is not found
            raise NotImplementedError
        except AttributeError:
            # Handle cases where the class is not found in the module
            raise NotImplementedError

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

# TODO: Fix this to be unified import for all environments
def customize_env(env_nickname, env):
    if env_nickname == 'pendulum':
        from envs_utils.gym.pendulum.pendulum_env_utils import pendulum_customize
        from envs_utils.gym.pendulum.pendulum_configs import env_config
        env = pendulum_customize(env)
        if env_config.use_wrapper:
            module = importlib.import_module('envs_utils.gym.pendulum.pendulum_env_utils')
            wrapper_cls = getattr(module, env_config.wrapper_name)
            return wrapper_cls(env)
        return env
    else:
        raise NotImplementedError
