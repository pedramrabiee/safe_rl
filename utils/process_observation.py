import numpy as np
import torch
from utils.torch_utils import get_tensor_blueprint
from utils.misc import isvec, torchify, np_object2dict
import importlib


class ObsProc:
    def __init__(self, env):
        """
        Initialize an observation processor.

        Args:
            env: The environment for which observation processing is done.
        """
        self.env = env
        self._proc_keys = []
        self._unproc_keys = []
        self.tensor_blueprint = None


    def initialize(self, init_dict=None):
        """
        Initialize the observation processor with optional parameters.

        Args:
            init_dict: Optional dictionary for initialization.
        """
        self._processors = {k: getattr(self, f'_proc_for_{k}') for k in self._proc_keys}
        self._unprocessors = {k: getattr(self, f'_unproc_from_{k}') for k in self._unproc_keys}
        self._obs_dims = {k: None for k in self._proc_keys}

    def obs_dim(self, proc_key=None):
        """
        Returns the observation dimension after applying processing.

        Args:
            proc_key: Key for the observation processor (optional).

        Returns:
            int: The observation dimension.
        """
        raise NotImplementedError

    def proc(self, obs, proc_key=None, proc_dict=None):
        """
        Process observations. Implement the env dependent processor in the _proc method
        _proc method act as a default processor:
        Case 1: self._proc_keys is NOT empty
           - In this case, if proc_key is provided, then proc_key should match one of the keys in self._proc_keys;
            otherwise, it throws an error
           - If you wish to use the default processor, do not provide the proc_key
         Case 2: self._proc_keys is empty
           - In this case, no matter what is the value of proc_key, the default processor is used

        Args:
            obs: The observations to be processed.
            proc_key: Key for the observation processor (optional).
            proc_dict: Optional dictionary for processing.

        Returns:
            Processed observations.
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
        """
        Reverse the observations processing.

        Args:
            obs: The observations to be reverse processing.
            proc_key: Key for the observation reverse processing (optional).
            unproc_dict: Optional dictionary for reverse processing.

        Returns:
            Reverse processed observations.
        """
        out = self._unproc(self._prep(obs), unproc_dict=unproc_dict)
        out = self._post_prep(out)
        return out

    def _proc(self, obs, proc_dict=None):
        """
        Default observation processor. To be implemented by derived classes.

        Args:
            obs: The observations to be processed.
            proc_dict: Optional dictionary for processing.

        Returns:
            Processed observations.
        """
        raise NotImplementedError

    def _unproc(self, obs, unproc_dict=None):
        """
        Default observation reverse processor. To be implemented by derived classes.

        Args:
            obs: The observations to be reverse processed.
            unproc_dict: Optional dictionary for reverse processing.

        Returns:
            Reverse processed observations.
        """
        raise NotImplementedError

    def _prep(self, obs):   # TODO: FIX THIS
        """
        Prepares observations for processing.

        Args:
            obs: The observations to be prepared.

        Returns:
            Prepared observations.
        """
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
        """
        Post-processes observations after processing.

        Args:
            obs: The observations to be post-prepped.

        Returns:
            Post-prepped observations.
        """
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
    env_collection = train_env['env_collection']
    nickname = train_env['env_nickname']

    parts = nickname.split('_')
    class_name = ''.join(part.capitalize() for part in parts)

    # Construct the module and class names
    module_name = f'envs_utils.{env_collection}.{nickname}.{nickname}_obs_proc'
    class_name = class_name + "ObsProc"

    try:
        obs_proc_module = importlib.import_module(module_name)
        obs_proc_cls = getattr(obs_proc_module, class_name)
        return obs_proc_cls
    except ImportError:
        # Handle cases where the module or class is not found
        return NeutralObsProc
    except AttributeError:
        # Handle cases where the class is not found in the module
        return NeutralObsProc
