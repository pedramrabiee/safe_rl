import importlib
from torch import is_tensor
import numpy as np


class CustomPlotter:
    """Class for custom plotting functionality for sampler"""
    def __init__(self, obs_proc):
        """
        Initialize plotter

        Args:
            obs_proc (ObservationProcessor): For processing observations before plotting
        """
        self.obs_proc = obs_proc
        self._data = dict()
        self._plot_schedule_by_itr = None
        self._plot_schedule_by_episode = None

    def push(self, data_dict=None):
        if not data_dict:
            return

        for k, v in data_dict.items():
            v = self._data_prep(k, v)
            if k in self._data:
                self._data[k] = np.vstack([self._data[k], v])
            if k not in self._data:
                self._data[k] = v

    def dump(self, itr=None, episode=None, dump_key=None, dump_dict=None):
        if itr is not None:
            self._dump_by_itr_or_episode(itr=itr, dump_dict=dump_dict)

        if episode is not None:
            self._dump_by_itr_or_episode(episode=episode, dump_dict=dump_dict)

        if dump_key is not None:
            for k in dump_key:
                assert hasattr(self, k)
                self._dump_by_key(k)

    def _dump_by_itr_or_episode(self, itr=None, episode=None, dump_dict=None):
        plt_sch = None
        counter = None
        if itr is not None:
            counter = itr
            plt_sch = self._plot_schedule_by_itr
            dump_dict = dict(**dump_dict, itr=itr) if dump_dict else dict(itr=itr)
            if itr == 0:
                if '0' in plt_sch:
                    for plot_key in plt_sch['0']:
                        self._dump_by_key(plot_key, dump_dict)
                    # Delete zero key so that it doesn't make problem in the following iterations
                    del self._plot_schedule_by_itr['0']
                return
        if episode is not None:
            counter = episode
            plt_sch = self._plot_schedule_by_episode
            dump_dict = dict(**dump_dict, episode=episode) if dump_dict else dict(episode=episode)

        if not plt_sch:
            return

        for k in plt_sch:
            if counter % int(k) == 0:
                for plot_key in plt_sch[k]:
                    self._dump_by_key(plot_key, dump_dict)

    def _dump_by_key(self, k, dump_dict=None):
        getattr(self, f'dump_{k}')(dump_dict)

    def _data_prep(self, data_key, data):
        if is_tensor(data):
            data = data.detach().numpy()
        data = np.atleast_2d(data)
        if hasattr(self, f'_prep_{data_key}'):
            prep_func = getattr(self, f'_prep_{data_key}')
            data = prep_func(data)
        return data

    def _make_data_array(self, data_key_config):
        data = []
        for dk in data_key_config:
            data.append(self._data[dk])
        return np.hstack(data)

    def _empty_data_by_key_list(self, data_key_list):
        if isinstance(data_key_list, list):
            for k in data_key_list:
                del self._data[k]
        else:
            del self._data[data_key_list]







def get_custom_plotter_cls(train_env):
    env_collection = train_env['env_collection']
    nickname = train_env['env_nickname']

    parts = nickname.split('_')
    class_name = ''.join(part.capitalize() for part in parts)

    # Construct the module and class names
    module_name = f'envs_utils.{env_collection}.{nickname}.{nickname}_plotter'
    class_name = class_name + "Plotter"

    try:
        custom_plotter_module = importlib.import_module(module_name)
        custom_plotter_cls = getattr(custom_plotter_module, class_name)
        return custom_plotter_cls
    except ImportError:
        # Handle cases where the module or class is not found
        return CustomPlotter
    except AttributeError:
        # Handle cases where the class is not found in the module
        return CustomPlotter





