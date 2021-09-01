import numpy as np
from attrdict import AttrDict
from utils.misc import torchify, add_noise


class ReplayBuffer:
    def __init__(self, max_size=int(100)):
        self._max_size = int(max_size)
        self._obs, self._next_obs, self._ac, self._rew, self._done, self._info = \
            None, None, None, None, None, None

    def push(self, experience):
        obs, ac, rew, next_obs, done, info = experience
        if isinstance(obs, list):
            obs = np.asarray(obs, dtype=object)
            next_obs = np.asarray(next_obs, dtype=object)
        info = np.asarray(info, dtype=object)

        if self._obs is None:
            self._obs = obs[-self._max_size:]
            self._ac = ac[-self._max_size:]
            self._rew = rew[-self._max_size:]
            self._next_obs = next_obs[-self._max_size:]
            self._done = done[-self._max_size:]
            self._info = info[-self._max_size:]
        else:
            self._obs = np.concatenate([self._obs, obs])[-self._max_size:]
            self._ac = np.concatenate([self._ac, ac])[-self._max_size:]
            self._rew = np.concatenate([self._rew, rew])[-self._max_size:]
            self._next_obs = np.concatenate([self._next_obs, next_obs])[-self._max_size:]
            self._done = np.concatenate([self._done, done])[-self._max_size:]
            self._info = np.concatenate([self._info, info])[-self._max_size:]

    @property
    def buffer_size(self):
        return self._rew.shape[0] if self._rew is not None else None

    def get_random_indices(self, batch_size):
        return np.random.choice(np.arange(self.buffer_size),
                                size=min(batch_size, self.buffer_size),
                                replace=False)

    def sample_by_indices(self, inds, device='cpu'):
        return AttrDict(obs=torchify(self._obs[inds], device=device),
                        ac=torchify(self._ac[inds], device=device),
                        rew=torchify(self._rew[inds], device=device),
                        next_obs=torchify(self._next_obs[inds], device=device),
                        done=torchify(self._done[inds], device=device),
                        info=self._info[inds])

    def sample(self, batch_size, device='cpu'):
        """return experience as torch tensor"""
        indices = self.get_random_indices(batch_size)
        return self.sample_by_indices(indices, device=device)

    @classmethod
    def set_buffer_size(cls, max_size):
        return cls(int(max_size))

    def get_stats(self, obs_preproc_func=None):
        # Preprocess observation if needed
        # TODO: check for modification for the dictionary observation
        obs = obs_preproc_func(self._obs) if obs_preproc_func else self._obs
        next_obs = obs_preproc_func(self._next_obs) if obs_preproc_func else self._next_obs
        return AttrDict(obs=AttrDict(mean=np.mean(obs, axis=0, dtype=np.float32),
                                     std=np.std(obs, axis=0, dtype=np.float32)),
                        ac=AttrDict(mean=np.mean(self._ac, axis=0, dtype=np.float32),
                                    std=np.std(self._ac, axis=0, dtype=np.float32)),
                        next_obs=AttrDict(mean=np.mean(next_obs, axis=0, dtype=np.float32),
                                          std=np.std(next_obs, axis=0, dtype=np.float32)),
                        delta_obs=AttrDict(mean=np.mean(next_obs - obs, axis=0, dtype=np.float32),
                                           std=np.std(next_obs - obs, axis=0, dtype=np.float32)))

    def get_buffer(self, to_tensor=False, device="cpu"):
        if to_tensor:
            return AttrDict(obs=torchify(self._obs, device=device),
                            ac=torchify(self._ac, device=device),
                            rew=torchify(self._rew, device=device),
                            next_obs=torchify(self._next_obs, device=device),
                            done=torchify(self._done, device=device),
                            info=self._info)
        else:
            return AttrDict(obs=self._obs,
                            ac=self._ac,
                            rew=self._rew,
                            next_obs=self._next_obs,
                            done=self._done,
                            info=self._info)


    def init_buffer(self, data):
        self._obs, self._ac, self._rew, self._next_obs, self._done, self_info = \
            data.obs, data.ac, data.rew, data.next_obs, data.done, data.info
