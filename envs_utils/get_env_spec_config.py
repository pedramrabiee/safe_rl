import importlib


def get_env_spec_config(train_env):
    collection = train_env['env_collection']
    nickname = train_env['env_nickname']

    module_name = f'envs_utils.{collection}.{nickname}.{nickname}_configs'

    try:
        env_config_module = importlib.import_module(module_name)
        env_config = getattr(env_config_module, "env_config")
        if collection == "safety_gym":
            engine_config = getattr(env_config_module, "engine_config")
            env_config.update(engine_config)
        return env_config
    except ImportError:
        raise NotImplementedError
    except AttributeError:
        raise NotImplementedError
