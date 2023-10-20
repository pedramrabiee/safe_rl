from utils.misc import scaler
import torch
import importlib

class NominalDynamics:
    def __init__(self, obs_dim, ac_dim, out_dim, timestep, env_bounds):
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.out_dim = out_dim
        self.env_bounds = env_bounds
        self.timestep = timestep

    def initialize(self, params, init_dict=None):
        pass

    def predict(self, obs, ac, split_return=False):
        ac = self._prep(ac)
        ac = scaler(
            ac,
            lim_from=(self.env_bounds.new.low, self.env_bounds.new.high),
            lim_to=(self.env_bounds.old.low, self.env_bounds.old.high)
        )      # TODO: this is only applicable to continuous action-space, fix this for discrete action-space
        out = self._predict(self._prep(obs), ac, split_return)
        if torch.is_tensor(obs):
            return torch.from_numpy(out)
        return out

    def _predict(self, obs, ac, split_return=False):
        raise NotImplementedError

    def _prep(self, x):
        if torch.is_tensor(x):
            x = x.numpy()
        return x


def get_nominal_dyn_cls(train_env, env):
    params = None

    env_collection = train_env['env_collection']
    nickname = train_env['env_nickname']

    parts = nickname.split('_')
    class_name = ''.join(part.capitalize() for part in parts)

    # Construct the module and class names
    module_name = f'envs_utils.{env_collection}.{nickname}.{nickname}_dynamics'
    class_name = class_name + "NominalDynamics"
    try:
        nom_dyn_module = importlib.import_module(module_name)
        nom_dyn_cls = getattr(nom_dyn_module, class_name)
        if env_collection == 'safety_gym':
            module_name = f'envs_utils.{env_collection}.{nickname}.{nickname}_utils'
            utils_module = importlib.import_module(module_name)
            params = getattr(utils_module, 'get_env_params')
        return nom_dyn_cls, params
    except ImportError:
        # Handle cases where the module or class is not found
        return NotImplementedError
    except AttributeError:
        # Handle cases where the class is not found in the module
        return NotImplementedError
