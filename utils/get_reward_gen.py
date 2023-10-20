import importlib


def get_reward_gen(train_env):
    env_collection = train_env['env_collection']
    nickname = train_env['env_nickname']

    # Construct the module and class names
    module_name = f'envs_utils.{env_collection}.{nickname}.{nickname}_reward_gen'
    func_name = nickname + "_reward_gen"

    try:
        reward_gen_module = importlib.import_module(module_name)
        reward_gen_func = getattr(reward_gen_module, func_name)
        return reward_gen_func
    except ImportError:
        # Handle cases where the module or class is not found
        raise 'Reward generator is not implemented for this environment.'
    except AttributeError:
        # Handle cases where the class is not found in the module
        raise 'Reward generator is not implemented for this environment.'
