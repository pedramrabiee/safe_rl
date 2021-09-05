import numpy as np
import torch
from utils.misc import e_and, e_not
from utils.torch_utils import apply_mask_to_dict_of_tensors


# TODO: add boolean for obs_proc, if the safe_set is defined based on the processed observation, then you need to use obs_prpoc,
#  otherwise, there is no need to use obs_proc

class SafeSet:
    def __init__(self, env, obs_proc):
        self.env = env
        self.obs_proc = obs_proc

    def safe_reset(self):
        raise NotImplementedError

    # safe action
    def get_safe_action(self, obs):
        raise NotImplementedError


class SafeSetFromData(SafeSet):
    def __init__(self, env, obs_proc):
        super().__init__(env, obs_proc)
        keys = ['in_safe', 'mid_safe', 'unsafe', 'out_cond_safe', 'out_cond_unsafe', 'mid_cond_safe']
        self.sets = {k: None for k in keys}
        self.makers = {k: getattr(self, 'make_' + k) for k in keys}

    def init_buffer(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.cursor = 0
        self.sets = {}

    def push_data(self, data):
        self.cursor += min(self.cursor + len(data), self.buffer_size)
        self.buffer[:0] = data  # append from beginning
        self.buffer = self.buffer[:self.buffer_size]

    def analyze_buffer(self):
        data = self.buffer[:self.cursor]
        for k in self.makers:
            self.update_sets(k, self.maker_loop(k, data))
        # reset cursor
        self.cursor = 0

    def maker_loop(self, key, data):
        samples = []
        for _ in range(data.robot_pos.shape[0]):
            samples.append(self.makers[key](data))
        return np.vstack(samples)

    def make_in_safe(self, data):
        raise NotImplementedError

    def make_mid_safe(self, data):
        raise NotImplementedError

    def make_unsafe(self, data):
        raise NotImplementedError

    def make_out_cond_safe(self, data):
        raise NotImplementedError

    def make_out_cond_unsafe(self, data):
        raise NotImplementedError

    def make_mid_cond_safe(self, data):
        raise NotImplementedError

    def update_sets(self, key, value):
        if self.sets[key]:
            self.sets[key] = np.concatenate((self.sets[key], value), axis=0)
        else:
            self.sets[key] = value

class SafeSetFromCriteria(SafeSet):

    def __init__(self, env, obs_proc):
        """
            Geometric safe set is comprised of three sections: inner, middle and outer
            Geometric unsafe set is the complement of the geometric safe set
        """
        super().__init__(env, obs_proc)

        keys = ['geo_safe', 'in_safe', 'mid_safe', 'unsafe', 'out_cond_safe', 'out_cond_unsafe', 'mid_cond_safe']
        self.criteria = {k: getattr(self, 'is_' + k) for k in keys}

    def sample_by_criteria(self, criteria_keys, batch_size):
        num_criteria = len(criteria_keys)
        samples = [[] for _ in range(num_criteria)]

        # repeat batch size for all criteria_keys if only one batch_size is given
        if isinstance(batch_size, list):
            assert len(batch_size) == len(criteria_keys), 'batch_size length does not match the criteria_keys length'
        else:
            batch_size = [batch_size for _ in range(num_criteria)]

        completed = [False for _ in range(num_criteria)]
        for k in criteria_keys:
            assert k in self.criteria, f'criterion {k} is not in criteria'

        while True:
            obs = self._get_obs()
            for i, k in enumerate(criteria_keys):
                if len(samples[i]) == batch_size[i]:
                    completed[i] = True
                if completed[i]:
                    continue
                if self.criteria[k](obs):
                    samples[i].append(obs)
                    break

            if all(completed):
                break

        return [np.vstack(sample) for sample in samples]


    def filter_sample_by_criteria(self, samples, criteria_keys):
        num_criteria = len(criteria_keys)
        masks = [None for _ in range(num_criteria)]

        for i, k in enumerate(criteria_keys):
            assert k in self.criteria, f'criterion {k} is not in criteria'
            masks[i] = self.criteria[k](samples)
        return masks if len(masks) > 1 else masks[0]

    def safe_reset(self):
        while True:
            obs = self.env.reset()
            if self.is_in_safe(self.obs_proc.proc(obs, proc_key='filter')):
                break
        return obs

    def is_geo_safe(self, obs):
        raise NotImplementedError

    def is_in_safe(self, obs):  # geometrically inside the inner safe section
        raise NotImplementedError

    def is_mid_safe(self, obs):  # geometrically inside the middle safe section
        raise NotImplementedError

    def is_out_safe(self, obs):  # geometrically inside the outer safe section
        raise NotImplementedError

    def is_unsafe(self, obs):
        raise NotImplementedError

    def is_ss_safe(self, obs):
        raise NotImplementedError

    def is_out_cond_safe(self, obs):  # conditionally safe in outer safe section
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return e_and(self.is_out_safe(obs), self.is_ss_safe(obs))

    def is_out_cond_unsafe(self, obs):  # conditionally unsafe in outer safe section
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return e_and(self.is_out_safe(obs), e_not(self.is_ss_safe(obs)))

    def is_mid_cond_safe(self, obs):  # conditionally safe in middle safe section
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return e_and(self.is_mid_safe(obs), e_not(self.is_ss_safe(obs)))

    def _get_obs(self):
        raise NotImplementedError


def get_safe_set(env_id, env, obs_proc, seed):
    safe_set = None
    if env_id == 'Pendulum-v0':
        from configs.env_configs.gym_envs.inverted_pendulum_configs import InvertedPendulumSafeSet
        safe_set = InvertedPendulumSafeSet(env, obs_proc)
        # set the Tuple seed
        safe_set.in_safe_set.seed(seed)
        safe_set.geo_safe_set.seed(seed)
    elif env_id == 'Point':
        # from configs.env_configs.safety_gym_envs.point_robot_configs import PointRobotSafeSetFromData
        from configs.env_configs.safety_gym_envs.point_robot_configs import PointRobotSafeSetFromCriteria
        safe_set = PointRobotSafeSetFromCriteria(env, obs_proc)
         # TODO: Set seeds if needed
    else:
        raise NotImplementedError
    return safe_set