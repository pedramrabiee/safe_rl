import importlib

class AffineInControlTorchDyn:
    def dynamics(self, state, u):
        return self.f(state) + self.g(state) * u

    def f(self, state):
        raise NotImplementedError

    def g(self, state):
        raise NotImplementedError




# TODO: make automatic torch dyn instantiator
def get_torch_dyn(env):
    env_collection = env['env_collection']
    nickname = env['env_nickname']

    parts = nickname.split('_')
    class_name = ''.join(part.capitalize() for part in parts)

    # Construct the module and class names
    module_name = f'envs_utils.{env_collection}.{nickname}.{nickname}_dynamics'
    class_name = class_name + "TorchDyn"

    try:
        torch_dyn_module = importlib.import_module(module_name)
        torch_dyn_cls = getattr(torch_dyn_module, class_name)
        return torch_dyn_cls
    except ImportError:
        # Handle cases where the module or class is not found
        raise ImportError('Module is not found')
    except AttributeError:
        # Handle cases where the class is not found in the module
        raise AttributeError('Class is not found')