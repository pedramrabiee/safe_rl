import numpy as np
from utils.process_observation import ObsProc


class PendulumObsProc(ObsProc):
    def obs_dim(self, proc_key=None):
        return int(3)

    def _proc(self, obs, proc_dict=None):
        return np.stack([np.arctan2(obs[..., 1], obs[..., 0]), obs[..., 2]], axis=-1)   # FIXME: fix this for ensemble and buffer

    def _unproc(self, obs, unproc_dict=None):
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)
        return np.stack([np.cos(obs[..., 0]), np.sin(obs[..., 0]), obs[..., 1]], axis=-1)            # FIXME: this will not work for buffer get_stats