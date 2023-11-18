import numpy as np
from utils.process_observation import ObsProc
import torch


obs_proc_index_dict = dict(
    rlbus=dict(module='pendulum_backup_shield', cls_name='PendulumObsProcBackupShield'),
    bus=dict(module='pendulum_backup_shield', cls_name='PendulumObsProcBackupShield'),
)


# Default Observation Processor, it is used if the agent key does not exist in obs_proc_index_dict

class PendulumObsProc(ObsProc):
    def __init__(self, env):
        super().__init__(env)
        self._proc_keys_indices = dict(
            safe_set='_trig_to_theta',
        )
        self._reverse_proc_dict = dict(
            _trig_to_theta='_theta_to_trig',
            _theta_to_trig='_trig_to_theta'
        )
        self._obs_dims = dict(_trig_to_theta=2,
                              _theta_to_trig=3)
    def _trig_to_theta(self, obs, proc_dict=None):
        if torch.is_tensor(obs):
            return torch.stack([torch.atan2(obs[..., 1], obs[..., 0]), obs[..., 2]], dim=-1)
        return np.stack([np.arctan2(obs[..., 1], obs[..., 0]), obs[..., 2]], axis=-1)


    def _theta_to_trig(self, obs, proc_dict=None):
        if torch.is_tensor(obs):
            obs = torch.atleast_1d(obs)
            return torch.stack([torch.cos(obs[..., 0]), torch.sin(obs[..., 0]), obs[..., 1]], dim=-1)
        # if obs.ndim == 1:
        #     obs = np.expand_dims(obs, axis=0)
        obs = np.atleast_1d(obs)
        return np.stack([np.cos(obs[..., 0]), np.sin(obs[..., 0]), obs[..., 1]], axis=-1)            # FIXME: this will not work for buffer get_stats


