import numpy as np
from utils.misc import np_object2dict
from collections import OrderedDict
from attrdict import AttrDict
from utils.misc import torchify
from buffers.replay_buffer import ReplayBuffer
from utils.seed import rng

class BufferQueue(ReplayBuffer):
    # init gets obs_proc, initializes attribute name strings as None, gets the pipeline
    def __init__(self, max_size=int(100)):
        super().__init__(max_size)
        self.reset_queue()

    def initialize(self, attribute_names, coupled_list=None):
        self.buffer = OrderedDict()
        for k in attribute_names:
            self.buffer[k] = None
        self.coupled_list = [[attribute_names.index(item) for item in coupled_set] for coupled_set in coupled_list]

    # def push(self, experience):
    #     obs, ac, rew, next_obs, done, info = experience
    #     experience = dict(obs=obs, ac=ac, rew=rew, next_obs=next_obs, done=done, info=info)
    #     self.push_to_queue(experience)

    # def push_to_queue(self, experience):
    #     self.queue.append(experience)

    def push_to_buffer(self, experience):
        """push experience to buffer, expect dictionary of numpy arrays"""
        for k in experience.keys():
            if experience[k] is None:
                continue
            if self.buffer[k] is None:
                self.buffer[k] = experience[k][-self._max_size:]
            else:
                self.buffer[k] = np.concatenate([self.buffer[k], experience[k]])[-self._max_size:]

    def reset_queue(self):
        self._obs, self._next_obs, self._ac, self._rew, self._done, self._info = \
            None, None, None, None, None, None

    def release_queue(self, to_tensor=False, device='cpu'):
        if to_tensor:
            out = AttrDict(obs=torchify(self._obs, device=device),
                           ac=torchify(self._ac, device=device),
                           rew=torchify(self._rew, device=device),
                           next_obs=torchify(self._next_obs, device=device),
                           done=torchify(self._done, device=device),
                           info=self._info)
        else:
            out = AttrDict(obs=self._obs,
                           ac=self._ac,
                           rew=self._rew,
                           next_obs=self._next_obs,
                           done=self._done,
                           info=self._info)
        self.reset_queue()
        return out

    @property
    def buffer_size(self):
        return [v.shape[0] if v is not None else None for _, v in self.buffer.items()]

    def get_random_indices(self, batch_size):
        buffer_sizes = self.buffer_size
        if not isinstance(batch_size, list):
            batch_size = [batch_size for _ in range(len(buffer_sizes))]

        indices = [rng.choice(np.arange(buffer_size),
                                    size=min(batch_size[i], buffer_size),
                                    replace=False) if buffer_size is not None else None
                   for i, buffer_size in enumerate(buffer_sizes)]
        if self.coupled_list:
            for coupled_set in self.coupled_list:
                for i in coupled_set:
                    indices[i] = indices[coupled_set[0]]
        return indices

    def sample_by_indices(self, inds, device='cpu'):
        return AttrDict({k: torchify(v[inds[i]], device=device) if inds[i] is not None else torchify(np.array([]), device=device)
                         for i, (k, v) in enumerate(self.buffer.items())})

    def get_buffer(self, to_tensor=False, device="cpu"):
        if to_tensor:
            raise NotImplementedError
        else:
            return self.buffer

    def init_buffer(self, data):
        self.buffer = data