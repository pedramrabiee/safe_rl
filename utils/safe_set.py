import numpy as np
import torch
from utils.misc import e_and, e_not, euler_integrator
from copy import copy


# TODO: add boolean for obs_proc, if the safe_set is defined based on the processed observation, then you need to use obs_prpoc,
#  otherwise, there is no need to use obs_proc

class SafeSet:
    def __init__(self, env, obs_proc):
        self.env = env
        self.obs_proc = obs_proc

    def initialize(self, init_dict=None):
        raise NotImplementedError

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
            Geometric safe set is comprised of two sections: inner and outer
            Geometric unsafe set is the complement of the geometric safe set
        """
        super().__init__(env, obs_proc)

        keys = ['geo_safe', 'in_safe', 'unsafe', 'out_cond_safe', 'out_cond_unsafe']

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

    # def is_mid_safe(self, obs):  # geometrically inside the middle safe section
    #     raise NotImplementedError

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

    # def is_mid_cond_safe(self, obs):  # conditionally safe in middle safe section
    #     if torch.is_tensor(obs):
    #         obs = obs.numpy()
    #     return e_and(self.is_mid_safe(obs), e_not(self.is_ss_safe(obs)))

    def _get_obs(self):
        raise NotImplementedError

class SafeSetFromPropagation(SafeSetFromCriteria):
    def __init__(self, env, obs_proc):
        """
            Geometric safe set is the desired positional safe set and it comprised of in_safe and in_unsafe
            Geometric unsafe set is the complement of the geometric safe set
        """
        super(SafeSetFromCriteria, self).__init__(env, obs_proc)
        keys = ['geo_safe', 'unsafe', 'in_safe', 'in_unsafe']
        self.criteria = {k: getattr(self, 'is_' + k) for k in keys}
        self.geo_safe_set = None
        self.max_T = 5
        self.dyn_pred = None

    def initialize(self, init_dict=None):
        self.dyn_pred = init_dict['dynamics_predictor']
        self.timestep = init_dict['timestep']

    def is_geo_safe(self, obs):
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return self.geo_safe_set.contains(self.obs_proc.proc(obs).squeeze()) # FIXME: you may need to modify obs_proc, you may want to also provide the key

    def is_in_safe(self, obs):
        if torch.is_tensor(obs):
            obs = obs.numpy()
        if np.isscalar(obs[0]):
            return self.is_geo_safe(obs) and self.forward_propagate(obs)
        return e_and(self.is_geo_safe(obs), self.forward_propagate(obs))

    def is_in_unsafe(self, obs):
        if torch.is_tensor(obs):
            obs = obs.numpy()
        if np.isscalar(obs[0]):
            return self.is_geo_safe(obs) and not(self.forward_propagate(obs))
        return e_and(self.is_geo_safe(obs), e_not(self.forward_propagate(obs)))

    def is_unsafe(self, obs):
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return e_not(self.is_geo_safe(obs))

    def forward_propagate(self, obs):
        if np.isscalar(obs[0]):
            obs = np.expand_dims(obs, axis=0)

        if obs.shape[0] > 1:
             return [self.forward_propagate(o) for o in obs]

        obs = self.obs_proc.proc(obs).squeeze() # TODO: check this
        is_safe = False
        x = copy(obs)
        for t in range(self.max_T):
            if self.is_unsafe(x):
                break
            ac = self.get_safe_action(x)
            # if t == 0:
            #     ac = -ac
            deriv_value = self.dyn_pred(x, ac, only_nominal=True, stats=None)
            new_x = self.compute_next_obs(deriv_value=deriv_value, obs=x)
            if self.propagation_terminate_cond(x, new_x):
                if self.is_geo_safe(new_x):
                    is_safe = True
                break
            x = new_x
        return is_safe

    def propagation_terminate_cond(self, obs, next_obs):
        raise NotImplementedError

    def compute_next_obs(self, deriv_value, obs):
        raise NotImplementedError

    def fast_criteria_check_for_single_obs(self, obs):
        pass


def get_safe_set(env_id, env, obs_proc, seed):
    safe_set = None
    if env_id == 'Pendulum-v0':
        # from envs_utils.gym.pendulum.pendulum_utils import InvertedPendulumSafeSet
        from envs_utils.gym.pendulum.pendulum_utils import InvertedPendulumSafeSetFromPropagation
        safe_set = InvertedPendulumSafeSetFromPropagation(env, obs_proc)
        # set the Tuple seed
        # safe_set.in_safe_set.seed(seed)
        safe_set.geo_safe_set.seed(seed)
    elif env_id == 'Point':
        # from envs.safety_gym.point_robot_utils import PointRobotSafeSetFromData
        from envs_utils.safety_gym.point_robot_utils import PointRobotSafeSetFromCriteria
        safe_set = PointRobotSafeSetFromCriteria(env, obs_proc)
         # TODO: Set seeds if needed
    elif env_id == 'cbf_test':
        from envs_utils.test_env.test_env_utils import CBFTestSafeSet, CBFTestSafeSetFromPropagation
        safe_set = CBFTestSafeSetFromPropagation(env, obs_proc)
        # safe_set.in_safe_set.seed(seed)
        safe_set.geo_safe_set.seed(seed)
    elif env_id == 'multi_mass_dashpot':
        from envs_utils.test_env.multi_m_dashpot_utils import MultiDashpotSafeSetFromPropagation
        safe_set = MultiDashpotSafeSetFromPropagation(env, obs_proc)
        safe_set.geo_safe_set.seed(seed)
    else:
        raise NotImplementedError
    return safe_set