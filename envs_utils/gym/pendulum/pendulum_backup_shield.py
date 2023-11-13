from utils.safe_set import SafeSetFromBarrierFunction
import torch
from math import pi
from attrdict import AttrDict
import numpy as np
from envs_utils.gym.pendulum.pendulum_configs import env_config


class PendulumBackupSet(SafeSetFromBarrierFunction):
    def initialize(self, init_dict=None):
        self.p = torch.tensor(init_dict.p)
        self.c = init_dict.c
        self.center = torch.tensor(init_dict.center)

    def safe_barrier(self, obs):
        # TODO: make obs and center matrix to handle batch
        result = torch.matmul(torch.matmul(obs - self.center, self.p), obs.T)
        return 1 - result / self.c if result.dim() == 0 else 1 - result.diag() / self.c


class PendulumSafeSet(SafeSetFromBarrierFunction):
    def initialize(self, init_dict=None):
        self.bounds = torch.tensor(init_dict.bounds).unsqueeze(dim=0)
        self.center = torch.tensor(init_dict.center).unsqueeze(dim=0)
        self.p_norm = init_dict.p_norm

    def late_initialize(self, init_dict=None):
        self.backup_agent = init_dict.backup_agent

    def des_safe_barrier(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs)
        return 1 - torch.norm((torch.atleast_1d(obs) - self.center) / self.bounds, p=self.p_norm, dim=1)

    def safe_barrier(self, obs):
        h, _, _ = self.backup_agent.get_h(obs)
        return h


class PendulumBackupControl:
    def __init__(self, gain, center, ac_lim):
        self.gain = torch.tensor(gain)
        self.center = torch.tensor(center)
        self.ac_lim = ac_lim

    def __call__(self, obs):
        return self.ac_lim * torch.tanh(torch.dot(self.gain, (obs - self.center)) / self.ac_lim)


_backup_sets_dict = dict(c=[0.17, 0.07, 0.07],
                         p=[
                             [[1.25, 0.25], [0.25, 0.25]],
                             [[1.3, 0.3], [0.3, 0.48]],
                             [[1.3, 0.3], [0.3, 0.48]]
                             ],
                         center=[0.0, pi/2, -pi/2]
)

_num_backup_sets_to_consider = 3
def get_backup_sets(env, obs_proc):
    backup_sets = [PendulumBackupSet(env, obs_proc) for _ in range(_num_backup_sets_to_consider)]
    for i, backup_set in enumerate(backup_sets):
        backup_set.initialize(init_dict=AttrDict(c=_backup_sets_dict['c'][i],
                                                 p=_backup_sets_dict['p'][i],
                                                 center=_backup_sets_dict['center'][i]))
    return backup_sets


_safe_set_dict = AttrDict(bounds=[pi, pi],
                          center=[0.0, 0.0],
                          p_norm=100)

def get_safe_set(env, obs_proc):
    safe_set = PendulumSafeSet(env, obs_proc)
    safe_set.initialize(init_dict=_safe_set_dict)
    return safe_set


# TODO: fix ac_lim
_backup_policies_dict = dict(
    gain=[
        [-3.0, -3.0],
        [-3.0, -3.0],
        [-3.0, -3.0]
    ],
    center=[
        [0.0, 0.0],
        [-pi/2, 0.0],
        [pi/2, 0.0],
    ],
    ac_lim=env_config.max_torque
)

def get_backup_policies():
    return [PendulumBackupControl(
        gain=_backup_policies_dict['gain'][i],
        center=_backup_policies_dict['center'][i],
        ac_lim=_backup_policies_dict['ac_lim']) for i in range(_num_backup_sets_to_consider)]

def desired_policy(obs):
    return np.array([0.0])

