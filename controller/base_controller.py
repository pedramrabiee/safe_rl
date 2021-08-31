from collections import namedtuple
from utils.misc import namedtuple2dict
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class BaseController:
    def __init__(self, dynamics, reward_gen, bounds, ac_dim, obs_proc=None):
        self.dynamics = dynamics
        self.reward_func = reward_gen(bounds)
        self.bounds = bounds
        self.ac_dim = ac_dim
        self.obs_proc = obs_proc

        if hasattr(self.dynamics, 'ensemble_size'):
            self.ensemble_size = self.dynamics.ensemble_size
        else:
            self.ensemble_size = None

    def initialize(self, params, init_dict):
        pass

    def act(self, obs, stats):
        """Either plan for the entire horizon and return the first contorl in the control sequence,
         or return single step control"""
        raise NotImplementedError

    def reset(self):
        pass

    def train(self):
        raise NotImplementedError

    def dream(self, obs, horizon, num_particles, stats, control_dict=None, get_entire_ac_seq=False):
        """forward simulation under the dynamics and with the controller for a horizon window"""
        if self.ensemble_size is None or self.ensemble_size == 1:
            obs = np.tile(obs, (num_particles, 1))
            pred_dict = None
            is_ensemble = False
        else:
            obs = np.tile(obs, (self.ensemble_size, num_particles, 1))
            pred_dict = dict(return_all=True)
            is_ensemble = True

        cache = []
        if get_entire_ac_seq:
            acs = self._control(obs, control_dict)
        for t in range(horizon):
            reward = np.zeros((*obs.shape[:-1], 1))
            if get_entire_ac_seq:
                ac = acs[:, t]
            else:
                ac = self._control(obs, control_dict)
            next_obs = self.dynamics.predict(obs, ac, stats, pred_dict=pred_dict)
            if is_ensemble:
                for i in range(self.ensemble_size):
                    reward[i] = self.reward_func(self.obs_proc.unproc(obs[i]) if self.obs_proc else obs[i],
                                                 ac[i])
            else:
                reward = self.reward_func(self.obs_proc.unproc(obs) if self.obs_proc else obs,
                                          ac)
            cache.append(Transition(state=obs, action=ac, next_state=next_obs, reward=reward))
            # revert observation to old format, since dynamics.predict method will call preprocess on observation anyway
            obs = self.obs_proc.unproc(next_obs) if self.obs_proc else next_obs
        return namedtuple2dict(cache)

    def _control(self, obs, control_dict=None):
        """return single step contorl"""
        raise NotImplementedError
