from utils.safe_set import SafeSetFromBarrierFunction
import numpy as np
import torch
from envs_utils.safety_gym.point.point_configs import env_config
from envs_utils.safety_gym.safety_gym_utils import SafetyGymSafeSetFromData, SafetyGymSafeSetFromCriteria
from utils import scale
from utils.seed import rng
from utils.torch_utils import softmin


class PointBackupSet(SafeSetFromBarrierFunction):
    def initialize(self, init_dict=None):
        pass


    def safe_barrier(self, obs):
        # TODO: compute region of attraction
        raise NotImplementedError



class PointSafeSet(SafeSetFromBarrierFunction):
    def initialize(self, init_dict=None):
        self.obstacle_pos = env_config.hazards_pos
        self.obstacle_radius = env_config.hazards_size
        self.p_norm = 2
        self.softmin_gain = 20

    def late_initialize(self, init_dict=None):
        self.backup_agent = init_dict.backup_agent

    def des_safe_barrier(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs)
        return softmin(torch.stack([torch.norm((torch.atleast_1d(obs) - obs_pos) / self.obstacle_radius - 1,
                                               p=self.p_norm, dim=1)
                                    for obs_pos in self.obstacle_pos]), self.softmin_gain)

    def safe_barrier(self, obs):
        h, _, _ = self.backup_agent.get_h(obs)
        return h

    def _get_obs(self):
        # TODO: implement this after getting and observation from safety gym and checking the states
        raise NotImplementedError


class PointBackupControl:
    def __init__(self):
        pass

    def __call__(self, obs):
        pass







