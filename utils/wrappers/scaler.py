import gym
from gym.spaces import Box
import numpy as np
from utils.misc import scaler
from attrdict import AttrDict

class ActionScalerWrapper(gym.Wrapper):
    def __init__(self, env, ac_lim=(-1.0, 1.0)):
        super().__init__(env)
        self.ac_lim = ac_lim

        assert isinstance(self.action_space,
                          Box), 'Only Box action space is currently implemented in the ActionScalerWrapper'

        # store the original observation space bounds
        if isinstance(self.action_space, Box):
            ac_shape = self.action_space.shape
            # store the original action space bounds
            self.ac_lim_old = (self.action_space.low, self.action_space.high)
            # update action space to new bounds
            self.action_space = Box(low=ac_lim[0], high=ac_lim[1], shape=ac_shape, dtype=np.float32)

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return observation, reward, np.array(done).astype(float), info

    def action(self, action):
        return scaler(action, lim_from=self.ac_lim, lim_to=self.ac_lim_old) if isinstance(self.action_space,
                                                                                          Box) else action
    def reverse_action(self, action):
        return scaler(action, lim_from=self.ac_lim_old, lim_to=self.ac_lim) if isinstance(self.action_space,
                                                                                          Box) else action
    @property
    def action_bounds(self):
        def tuple2dict(bounds):
            return AttrDict(low=bounds[0], high=bounds[1])

        return AttrDict(old=tuple2dict(self.ac_lim_old),
                        new=tuple2dict(self.ac_lim))
