import numpy as np
import torch
from utils.torch_utils import get_tensor_blueprint
from utils.misc import isvec, torchify, np_object2dict
import importlib

# Guide on implementing observation processors:
# - Base the observation on the environment's observation.
# - ObsProc includes methods: obs_dim, proc, _prep, _post_prep.
# - Subclasses should overwrite the _proc_keys_indices dictionary.
# - _proc_keys_indices maps keys to processors, specifying the processor for each key.
#   Example: _proc_keys_indices = {'mf': 'model_free_processor',
#                                'backup_set': 'trig_to_theta_processor',
#                                'backup_control': 'trig_to_theta_processor'}

# Methods in ObsProc:
# - obs_dim: Returns the environment's observation dimension if proc_key is None.
# - obs_dim with proc_key not None: Returns the processed dimension for a given key;
# if the key doesn't exist, it calls obs_dim with proc_key=None
# - proc: Processes observation based on a processor key (proc_key).
# - If the proc_key exists in _proc_keys_indices,
#   use the corresponding processor (i.e., _proc_keys_indices[proc_key]).
# - If proc_key does not exist in _proc_keys_indices, issue a warning and use the default proc behavior.
# - The default proc behavior is to perform preparation and no further processing.



class ObsProc:
    def __init__(self, env):
        """
        Initialize an observation processor.

        Args:
            env: The environment for which observation processing is done.
        """
        self.env = env
        self._proc_keys_indices = dict()    # Implement in subclasses
        self._obs_dims = dict()             # Implement in subclasses
        self._reverse_proc_dict = dict()    # Implement in subclasses
        self.tensor_blueprint = None

    def initialize(self, init_dict=None):
        """
        Initialize the observation processor with optional parameters.

        Args:
            init_dict: Optional dictionary for initialization.
        """
        # self._processors = {k: getattr(self, f'_proc_for_{k}') for k in self._proc_keys}
        # self._unprocessors = {k: getattr(self, f'_unproc_from_{k}') for k in self._unproc_keys}
        # self._obs_dims = {k: None for k in self._processors}

    def obs_dim(self, proc_key=None):
        """
        Returns the observation dimension after applying processing.

        Args:
            proc_key: Key for the observation processor (optional).

        Returns:
            int: The observation dimension.
        """

        # TODO: Currently just support one-dimensional Box observation spaces.
        if proc_key in self._proc_keys_indices:
            return self._obs_dims[self._proc_keys_indices[proc_key]]
        return self.env.observation_space.shape[0]


    def proc(self, obs, proc_key=None, proc_dict=None, reverse=False):
        proc = self._proc
        if proc_key and self._proc_keys_indices:    # if self._proc_keys is not None and proc_key is not empty
            # assert proc_key in self._proc_keys, 'processor key is invalid'
            if proc_key in self._proc_keys_indices:
                if not reverse:
                    proc = getattr(self, self._proc_keys_indices[proc_key])
                else:
                    proc = getattr(self, self._reverse_proc_dict[self._proc_keys_indices[proc_key]])

        # 2. reset tensor_blueprint back to None
        # tensor_blueprint is populated by the _prep function to the dtype and device of the torch tensors in the observation
        # if it finds the datatype of observations are torch tensor
        # self.tensor_blueprint = None

        # 3. Call processor on observation
        # out = proc(self._prep(obs), proc_dict=proc_dict)
        # out = proc(self._prep(obs), proc_dict=proc_dict)


        # 4. Call _post_prep on the processed observation
        # call _prep again to expand dims if needed and to torchify arrays if needed
        # out = self._post_prep(out)
        return proc(obs, proc_dict=proc_dict)

    def _proc(self, obs, proc_dict=None):
        """
        Default observation processor. To be implemented by derived classes.

        Args:
            obs: The observations to be processed.
            proc_dict: Optional dictionary for processing.

        Returns:
            Processed observations.
        """
        return obs

    # def _prep(self, obs):   # TODO: FIX THIS
    #     """
    #     Prepares observations for processing.
    #
    #     Args:
    #         obs: The observations to be prepared.
    #
    #     Returns:
    #         Prepared observations.
    #     """
    #     # if torch.is_tensor(obs):
    #     #     if not self.tensor_blueprint: # only once we store tensor blueprint (assumes all the observations have the
    #     #         # same dtype and device and requires grad)
    #     #         self.tensor_blueprint = get_tensor_blueprint(obs)
    #     #     return self._prep(obs.numpy())
    #     # if isinstance(obs, np.ndarray) and not isinstance(obs[0], dict):
    #     #     return obs.reshape(1, -1) if isvec(obs) else obs
    #     # if isinstance(obs, dict):
    #     #     return {k: self._prep(v) for k, v in obs.items()}
    #     # if isinstance(obs, np.ndarray) and isinstance(obs[0], dict):
    #     #     # vstack (torch or np) items with the same key and return a dictionary with the same keys
    #     #     obs = np_object2dict(obs, ret_AttrDict=False)
    #     #     out = self._prep(obs)
    #     #     return out
    #     return obs


    # def _post_prep(self, obs):
    #     """
    #     Post-processes observations after processing.
    #
    #     Args:
    #         obs: The observations to be post-prepped.
    #
    #     Returns:
    #         Post-prepped observations.
    #     """
    #     obs = self._prep(obs)       # expand numpy array dims if needed
    #     out = torchify(obs, **self.tensor_blueprint) if self.tensor_blueprint else obs
    #     return out



# class NeutralObsProc(ObsProc):
#     """Neutral Observation Processes: This class is used when do_obs_proc is False"""
#     def obs_dim(self, proc_key=None):
#         return self.env.observation_space.shape[0]
#
#     def proc(self, obs, proc_key=None, proc_dict=None):
#         self.tensor_blueprint = None
#         return self._post_prep(self._prep(obs))
#
#     def unproc(self, obs, proc_key=None, unproc_dict=None):
#         self.tensor_blueprint = None
#         return self._post_prep(self._prep(obs))


def get_obsproc_cls(train_env, agent):
    env_collection = train_env['env_collection']
    nickname = train_env['env_nickname']

    parts = nickname.split('_')
    class_name = ''.join(part.capitalize() for part in parts)

    # Construct the module and class names
    module_name = f'envs_utils.{env_collection}.{nickname}.{nickname}_obs_proc'

    class_name = class_name + "ObsProc"

    try:
        obs_proc_module = importlib.import_module(module_name)
        index_dict = getattr(obs_proc_module, 'obs_proc_index_dict')
        if agent in index_dict:
            obs_proc_module = index_dict[agent]['module']
            class_name = index_dict[agent]['cls_name']
            obs_proc_module = importlib.import_module(obs_proc_module)
            obs_proc_cls = getattr(obs_proc_module, class_name)
        elif hasattr(obs_proc_module, class_name):
            obs_proc_cls = getattr(obs_proc_module, class_name)
        else:
            print('NeutralObsProc is used as observation processor')
            obs_proc_cls = ObsProc
        return obs_proc_cls
    except ImportError:
        print('Import Error: NeutralObsProc is used as observation processor')
        # Handle cases where the module or class is not found
        return ObsProc
    except AttributeError:
        print('Attribute Error: NeutralObsProc is used as observation processor')
        # Handle cases where the class is not found in the module
        return ObsProc


