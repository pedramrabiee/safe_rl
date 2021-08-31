import gym
from gym.spaces import Box
import numpy as np
from utils.misc import scaler
from attrdict import AttrDict

class ActionScalerWrapper(gym.Wrapper):
    def __init__(self, env, obs_lim=None, ac_lim=(-1.0, 1.0)):
        super().__init__(env)
        self.ac_lim = ac_lim

        assert isinstance(self.observation_space,
                          Box), 'Only Box observation space is currently implemented in the ActionScalerWrapper'
        assert isinstance(self.action_space,
                          Box), 'Only Box action space is currently implemented in the ActionScalerWrapper'


        # store the original observation space bounds
        if isinstance(self.observation_space, Box):
            self.obs_lim_old = (self.observation_space.low, self.observation_space.high)
            if obs_lim:
                self.is_obs_scaled = True
                obs_shape = self.observation_space.shape
                self.obs_lim = obs_lim
                # update observation space to new bounds
                self.observation_space = Box(low=obs_lim[0], high=obs_lim[1], shape=obs_shape, dtype=np.float32)
            else:
                self.is_obs_scaled = False
                self.obs_lim = self.obs_lim_old

        if isinstance(self.action_space, Box):
            ac_shape = self.action_space.shape
            self.ac_lim_old = (self.action_space.low, self.action_space.high)
            # store the original action space bounds
            self.ac_lim_old = (self.action_space.low, self.action_space.high)
            # update action space to new bounds
            self.action_space = Box(low=ac_lim[0], high=ac_lim[1], shape=ac_shape, dtype=np.float32)


    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return self.observation(observation), reward, np.array(done).astype(float), info

    def observation(self, observation):
        return scaler(observation, lim_from=self.obs_lim_old, lim_to=self.obs_lim) if self.is_obs_scaled else observation


    def action(self, action):
        return scaler(action, lim_from=self.ac_lim, lim_to=self.ac_lim_old) if isinstance(self.action_space,
                                                                                          Box) else action

    def reverse_action(self, action):
        return scaler(action, lim_from=self.ac_lim_old, lim_to=self.ac_lim) if isinstance(self.action_space,
                                                                                          Box) else action

    @property
    def bounds(self):
        # TODO: this only works for continuous action, add discrete action
        def _bounds_tup2dict(bounds):
            return AttrDict(low=bounds[0], high=bounds[1])

        return AttrDict(old=AttrDict(obs=_bounds_tup2dict(self.obs_lim_old),
                                     ac=_bounds_tup2dict(self.ac_lim_old)),
                        new=AttrDict(obs=_bounds_tup2dict(self.obs_lim),
                                     ac=_bounds_tup2dict(self.ac_lim)))
