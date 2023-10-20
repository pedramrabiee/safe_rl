import importlib


class CustomPlotter:
    """Class for custom plotting functionality for sampler"""
    def __init__(self, obs_proc):
        """
        Initialize plotter

        Args:
            obs_proc (ObservationProcessor): For processing observations before plotting
        """
        self.obs_proc = obs_proc

    def sampler_push_obs(self, obs):

        """
        Add observation to plot

        Args:
            obs (np.ndarray): Observation
        """
        pass

    def filter_push_action(self, ac):
        """
        Add action to plot

        Args:
            ac (np.ndarray): Action
        """
        pass


    def dump_sampler_plots(self, episode_num):
        """
        Save sampler plots for given episode

        Args:
            episode_num (int): Episode number
        """
        pass


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





