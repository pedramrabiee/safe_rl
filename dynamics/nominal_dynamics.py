import numpy as np
from utils.misc import scaler
import torch

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


def get_nominal_dyn_cls(train_env):
    if train_env['env_collection'] == 'gym':
        if train_env['env_id'] == 'Pendulum-v0':
            from configs.env_configs.gym_envs.inverted_pendulum_configs import InvertedPendulumNominalDynV2
            params = None
            return InvertedPendulumNominalDynV2, params
        else:
            raise NotImplementedError
    elif train_env['env_collection'] == 'safety_gym':
        if train_env['env_id'] == 'Point':
            from configs.env_configs.safety_gym_envs.point_robot_configs import PointRobotNominalDynamics
            params = None
            return PointRobotNominalDynamics, params
    else:
        raise NotImplementedError
