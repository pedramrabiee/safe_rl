import numpy as np
import torch
from utils.torch_utils import get_tensor_blueprint
from utils.misc import isvec, torchify, np_object2dict


class ObsProc:
    def __init__(self, env):
        self.env = env
        self._proc_keys = []
        self._unproc_keys = []
        self.tensor_blueprint = None


    def initialize(self, init_dict=None):
        self._processors = {k: getattr(self, f'_proc_for_{k}') for k in self._proc_keys}
        self._unprocessors = {k: getattr(self, f'_unproc_from_{k}') for k in self._unproc_keys}
        self._obs_dims = {k: None for k in self._proc_keys}

    def obs_dim(self, proc_key=None):
        """returns observation dimension after applying proc"""
        raise NotImplementedError

    def proc(self, obs, proc_key=None, proc_dict=None):
        """
        Processes observation. Implement the env dependent processor in the _proc method
        _proc method act as a default processor:
        Case 1: self._proc_keys is NOT empty
           - In this case, if proc_key is provided, then proc_key should match one of the keys in self._proc_keys;
            otherwise, it throws an error
           - If you wish to use the default processor, do not provide the proc_key
         Case 2: self._proc_keys is empty
           - In this case, no matter what is the value of proc_key, the default processor is used
        """

        if proc_key and self._proc_keys:    # if self._proc_keys is not None and proc_key is not empty
            assert proc_key in self._proc_keys, 'processor key is invalid'
            proc = self._processors[proc_key]
        else:   # if proc_key is provided but self._proc_keys is None or vice versa
            proc = self._proc
        # reset tensor_blueprint back to None
        # tensor_blueprint is populated by the _prep function to the dtype and device of the torch tensors in the observation
        # if it finds the datatype of observations are torch tensor
        self.tensor_blueprint = None

        out = proc(self._prep(obs), proc_dict=proc_dict)
        # call _prep again to expand dims if needed and to torchify arrays if needed
        out = self._post_prep(out)
        return out

    def unproc(self, obs, proc_key=None, unproc_dict=None):
        """Unprocess observation. Implement the env dependent unprocessor in the _unproc method"""
        out = self._unproc(self._prep(obs), unproc_dict=unproc_dict)
        out = self._post_prep(out)
        return out

    def _proc(self, obs, proc_dict=None):
        raise NotImplementedError

    def _unproc(self, obs, unproc_dict=None):
        raise NotImplementedError

    def _prep(self, obs):   # TODO: FIX THIS
        if torch.is_tensor(obs):
            if not self.tensor_blueprint: # only once we store tensor blueprint (assumes all the observations have the
                # same dtype and device and requires grad)
                self.tensor_blueprint = get_tensor_blueprint(obs)
            return self._prep(obs.numpy())
        if isinstance(obs, np.ndarray) and not isinstance(obs[0], dict):
            return obs.reshape(1, -1) if isvec(obs) else obs
        if isinstance(obs, dict):
            return {k: self._prep(v) for k, v in obs.items()}
        if isinstance(obs, np.ndarray) and isinstance(obs[0], dict):
            # vstack (torch or np) items with the same key and return a dictionary with the same keys
            obs = np_object2dict(obs, ret_AttrDict=False)
            out = self._prep(obs)
            return out


    def _post_prep(self, obs):
        obs = self._prep(obs)       # expand numpy array dims if needed
        out = torchify(obs, **self.tensor_blueprint) if self.tensor_blueprint else obs
        return out



class NeutralObsProc(ObsProc):
    """Neutral Observation Processes: This class is used when do_obs_proc is False"""
    def obs_dim(self, proc_key=None):
        return self.env.observation_space.shape[0]

    def proc(self, obs, proc_key=None, proc_dict=None):
        self.tensor_blueprint = None
        return self._post_prep(self._prep(obs))

    def unproc(self, obs, proc_key=None, unproc_dict=None):
        self.tensor_blueprint = None
        return self._post_prep(self._prep(obs))


def get_obsproc_cls(train_env):
    if train_env['env_collection'] == 'gym':
        if train_env['env_id'] == 'Pendulum-v0':
            from envs_utils.gym.pendulum.pendulum_utils import InvertedPendulumObsProc
            return InvertedPendulumObsProc
        else:
            raise NotImplementedError
    elif train_env['env_collection'] == 'safety_gym':
        if train_env['env_id'] == 'Point':
            from envs_utils.safety_gym.point_robot_utils import PointObsProc
            return PointObsProc
    else:
        raise NotImplementedError


