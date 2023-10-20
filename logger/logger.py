import wandb
import numpy as np
import os.path as osp
import os
from utils.console import colorize
from utils.misc import get_timestamp
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.message import EmailMessage
from pandas import DataFrame
import json

# Setting some configuration for matplotlib
mpl.rcParams['agg.path.chunksize'] = 10000
mpl.rcParams['text.usetex'] = True

_sns_colormap = 'husl'  # Define the seaborn colormap. Other options, husl, rocket, Paired, tab10
_colormap = 'jet'       # Define the matplotlib colormap
_config = None

# Initialize string variables
_root_dir = ''      # Root directory
logdir = ''         # Log directory
plotdir = ''        # Plot directory

# Initialize dictionaries and variables for logging
_tabular_fds = {}       # File descriptors for tabular data
_tabular_headers = {}   # Headers for tabular data
_tabular_queue = {}      # Queue for tabular data
_steps = {}              # Dictionary to keep track of steps
_plot_queue = {}         # Queue for plotting data
_plotter_in_eval_mode = False   # Flag to indicate if the plotter is in evaluation mode



def initialize(config, root_dir):
    """
     Initialize the logging and configuration.

     Args:
         config: Configuration object.
         root_dir: Root directory for logging and results.
     """
    global _config

    _config = config
    # Initialize wandb
    if not _config.debugging_mode:
        _initialize_wandb(root_dir)
    # Setup CSV and tabular data
    _initialize_tabular()
    # Create a plot directory
    _create_plot_dir()

def _initialize_wandb(root_dir):
    """
    Initialize Weights and Biases (wandb) for logging.

    Args:
        root_dir: Root directory for logging and results.
    """
    global logdir

    # Define the wandb directory path
    wandb_dir = osp.join(root_dir, _config.results_dir, _config.wandb_project_name)
    os.makedirs(wandb_dir, exist_ok=True)
    if _config.resume:
        resume = "must"
        id = _config.load_run_id
    else:
        resume = "never"
        id = wandb.util.generate_id()

    # Log in to wandb, initialize the run, and set the project directory
    wandb.login()
    wandb.init(id=id,
               config=_config,
               project=_config.wandb_project_name,
               dir=wandb_dir,
               resume=resume)
    logdir = wandb.run.dir

def _initialize_tabular():
    """
        Initialize tabular logging, including CSV files and related data structures.

        This function sets up CSV files for logging various tabular data during the experiment.
        It creates directories and initializes data structures needed for logging.

        Global variables affected:
        - _tabular_fds: File descriptors for writing and reading CSV files.
        - _tabular_headers: CSV file headers.
        - _tabular_queue: Data queue for tabular logging.
        - _steps: Counters for each type of tabular data.

        Global directories:
        - Creates directories for storing tabular data within the project's log directory.
        """
    global _tabular_fds, _tabular_headers, _tabular_queue, _steps

    # Define CSV file names to be used for different types of tabular data
    csv_file_names = ['episode', 'evaluation_episode', 'iteration', 'epoch', 'cbf_epoch', 'dynamics_epoch']

    # Create a subdirectory within the log directory to store tabular data
    tabular_dir = osp.join(logdir, 'tabular')
    if not _config.debugging_mode:
        os.makedirs(tabular_dir, exist_ok=True)

        # Generate paths for the CSV files
        csv_dir = [osp.join(tabular_dir, file + '_progress.csv') for file in csv_file_names]

        # Open file descriptors for writing and reading for each CSV file
        _tabular_fds['w'] = {k: open(file, mode='w') for k, file in zip(csv_file_names, csv_dir)}
        _tabular_fds['r'] = {k: open(file, mode='r') for k, file in zip(csv_file_names, csv_dir)}

    # Initialize tabular data headers
    _tabular_headers = {k: None for k in csv_file_names}

    # Create an empty queue for each CSV file type
    _tabular_queue = {k: {} for k in csv_file_names}

    # Initialize counters for each type of tabular data
    # Initialize counters for each type of tabular data
    _steps = {k: 0 for k in csv_file_names}


def _create_plot_dir():
    """
    Create a directory for storing plots and figures.

    This function sets up a directory within the project's log directory for storing generated plots and figures.

    Global variables affected:
    - plotdir: The directory path for storing plots.
    """
    global plotdir

    # Create a subdirectory within the log directory for storing plots
    plotdir = osp.join(logdir, 'plots')
    os.makedirs(plotdir, exist_ok=True)


def log(msg, color='green'):
    """
    Print a colorized message to the standard output (stdout).

    Args:
    - msg (str): The message to be printed.
    - color (str): The color used to format the message (default is 'green').

    Global variables used:
    - _config: A global configuration object.

    This function prints a message to the standard output (stdout) with optional color formatting.
    If the `_config.add_timestamp` flag is set, it prepends a timestamp to the message before printing it.

    Example:
    log("Hello, World!", color='red')

    Output:
    [Timestamp] Hello, World!
    """

    if _config.add_timestamp:
        msg = get_timestamp() + msg
    print(colorize(msg, color, bold=True))


def push_tabular(data, cat_key):
    """
        Add a list of dictionaries to the tabular queue for later use in the 'dump_csv' function.

        Args:
        - data: A list of dictionaries containing data to be added to the tabular queue.
        - cat_key: The category key under which the data should be stored.

        Global variables used:
        - _tabular_queue: A dictionary that stores tabular data queues categorized by keys.

        Use this method when you have a list of dictionaries that need to be stored in the tabular queue
        for later use with the 'dump_csv' function.

        Example:
        data = [{'episode': 1, 'score': 100}, {'episode': 2, 'score': 150}]
        cat_key = 'iteration'
        push_tabular(data, cat_key)
        """
    global _tabular_queue
    """use this method when you have a list of dictionary to be added to _tabular_queue for later use in dump_csv"""
    _tabular_queue[cat_key] = data


def add_tabular(data_dict, cat_key='iteration', stats=False, for_csv=False):
    """
        Add data to the tabular queue and log it in tabular format.

        Args:
        - data_dict: A dictionary containing the data to be added to the tabular queue.
        - cat_key: The category key for organizing the data in the tabular queue.
        - stats: A boolean indicating whether to compute and log statistics (mean, std, max, min) for the data.
        - for_csv: A boolean indicating whether the data should be logged to a CSV file.

        Global variables used:
        - _tabular_queue: A dictionary that stores tabular data queues categorized by keys.
        - _config: A global configuration object.

        This function adds data to the tabular queue under the specified category key.
        If 'stats' is set to True, it also computes statistics for the data and logs them.
        If 'for_csv' is True, the data is also logged to a CSV file.

        Example:
        data = {'iteration': 100, 'reward': 25.0}
        cat_key = 'iteration'
        add_tabular(data, cat_key)
        """
    if stats:
        for k, v in data_dict.items():
            stats = add_stats(k, v)
            _log_tabular(stats, cat_key)
    else:
        _log_tabular(data_dict, cat_key)


def _log_tabular(data_dict, cat_key):
    """
        Log data to the tabular queue under a specified category key.

        Args:
        - data_dict: A dictionary containing the data to be logged.
        - cat_key: The category key for organizing the data in the tabular queue.

        Global variables used:
        - _tabular_queue: A dictionary that stores tabular data queues categorized by keys.

        This function logs data to the tabular queue under the specified category key.
         It iterates through the key-value pairs in the provided data dictionary and stores them in the queue.

        Example:
        data = {'reward': 25.0, 'steps': 100}
        cat_key = 'iteration'
        _log_tabular(data, cat_key)
        """
    global _tabular_queue

    for k, v in data_dict.items():
        _tabular_queue[cat_key][k] = v


def dump_tabular(cat_key='iteration', log=False, wandb_log=False, csv_log=False):
    """
    Dump tabular data to various log formats and clear the tabular queue.

    Args:
    - cat_key: The category key for the tabular data (default is 'iteration').
    - log: Whether to log data to the console (default is False).
    - wandb_log: Whether to log data to Weights & Biases (WandB) (default is False).
    - csv_log: Whether to log data to a CSV file (default is False).

    Global variables used:
    - _config: Configuration settings.
    - _tabular_queue: A dictionary that stores tabular data queues categorized by keys.
    - _steps: A dictionary that keeps track of steps for different categories.

    This function is used to dump tabular data to various log formats and clear the tabular queue.
    It takes several arguments, including the category key, a flag to log data to the console,
    a flag to log data to WandB, and a flag to log data to a CSV file. If the `log` flag is set to True,
    the data is logged to the console using the `dump_console` function.
    If the `wandb_log` flag is set to True and WandB is enabled, the data is logged to WandB using the
    `dump_wandb` function. If the `csv_log` flag is set to True, the data is logged to a CSV file using the
    `dump_csv` function. After logging the data, the tabular queue for the specified category key is cleared
    using the `clear_tabular_queue` function.

    Example:
    cat_key = 'iteration'
    log = True
    wandb_log = True
    csv_log = True
    dump_tabular(cat_key, log, wandb_log, csv_log)
    """
    global _steps

    if _config.debugging_mode:
        return

    queue = _tabular_queue[cat_key]
    _steps[cat_key] += 1
    queue[cat_key] = _steps[cat_key]
    if log:
        dump_console(queue)
    if wandb_log and _config.use_wandb:
        dump_wandb(queue)
    if csv_log:
        dump_csv(cat_key)
    clear_tabular_queue(cat_key)


def dump_console(queue):
    """
    Print tabular data to the console.

    Args:
    - queue: A dictionary containing tabular data to be printed.

    Global variables used:
    - _config: Configuration settings.

    This function is used to print tabular data to the console. It takes a dictionary 'queue' as an argument,
    which contains the tabular data to be printed. The function formats the data for printing,
    including aligning columns, and prints it to the console. If the data values are floating-point numbers,
    they are formatted to have a specific width and precision.

    Example:
    queue = {'Step': 100, 'Loss': 0.123, 'Reward': 100.0}
    dump_console(queue)
    """
    vals = []
    key_lens = [len(key) for key in queue]
    max_key_len = max(15, max(key_lens))
    keystr = '%' + '%d' % max_key_len
    fmt = "| " + keystr + "s | %15s |"
    n_slashes = 22 + max_key_len
    print("-" * n_slashes)
    for key in queue:
        val = queue.get(key, "")
        valstr = "%8.3g" % val if hasattr(val, "__float__") else val
        print(fmt % (key, valstr))
        vals.append(val)
    print("-" * n_slashes, flush=True)


def dump_wandb(queue):
    """
    Log tabular data to Weights and Biases (wandb).

    Args:
    - queue: A dictionary containing tabular data to be logged.

    Global variables used:
    - wandb: The Weights and Biases Python library.

    This function logs tabular data to Weights and Biases (wandb) using the provided dictionary 'queue'.
    It allows for tracking and visualizing experiment metrics in the wandb dashboard.

    Example:
    queue = {'Step': 100, 'Loss': 0.123, 'Reward': 100.0}
    dump_wandb(queue)
    """
    wandb.log(queue)


def push_plot(data, plt_key, row_append=False):
    """
        Push data to a plot queue for later visualization.

        Args:
        - data: The data to be added to the plot queue.
        - plt_key: The key identifying the specific plot in the queue.
        - row_append: If True, append data to the last row; otherwise, create a new row in the plot queue.

        Global variables used:
        - _plot_queue: A dictionary containing plot data.
        - _plotter_in_eval_mode: A flag indicating whether the plotter is in evaluation mode.
        - _config: The global configuration settings.

        This function is used to push data to a plot queue, allowing for later visualization of custom plots.
        The 'plt_key' is used to identify the specific plot within the queue.
        The 'row_append' parameter controls whether the data is appended to the last row or if a new row is created
        in the plot queue.

        Example:
        data = np.array([1, 2, 3, 4])
        plt_key = 'custom_plot'
        push_plot(data, plt_key, row_append=True)
        """
    global _plot_queue, _plotter_in_eval_mode, _config

    if not _config.plot_custom_figs or _plotter_in_eval_mode:
        return

    if plt_key in _plot_queue.keys():
        if row_append:
            _plot_queue[plt_key][-1] = np.hstack([_plot_queue[plt_key][-1], data])
        else:
            _plot_queue[plt_key].append(data)
    else:
        _plot_queue[plt_key] = []
        _plot_queue[plt_key].append(data)


def set_plotter_in_eval_mode():
    """
    Set the plotter in evaluation mode.

    Global variables used:
    - _plotter_in_eval_mode: A flag indicating whether the plotter is in evaluation mode.

    This function sets the '_plotter_in_eval_mode' flag to True, indicating that the plotter is in evaluation mode.
    In this mode, the plotter may behave differently or restrict certain actions related to plotting.
    """
    global _plotter_in_eval_mode
    _plotter_in_eval_mode = True


def set_plotter_in_train_mode():
    """
    Set the plotter in training mode.

    Global variables used:
    - _plotter_in_eval_mode: A flag indicating whether the plotter is in evaluation mode.

    This function sets the '_plotter_in_eval_mode' flag to False, indicating that the plotter is in training mode.
    In this mode, the plotter should operate as usual and perform actions relevant to training.
    """
    global _plotter_in_eval_mode
    _plotter_in_eval_mode = False


def dump_plot(filename, plt_key, step_key=None):
    """
    Save a plot as an image file and display it. Optionally, log it to WandB.

    Parameters:
    - filename (str): The filename for the saved image.
    - plt_key (str): A key associated with the plot.
    - step_key (str, optional): A key for tracking the step.

    Global variables used:
    - plotdir: The directory where plot images are saved.
    - _config: A global configuration object containing settings.
    - plt: The Matplotlib library for generating plots.
    - _steps: A dictionary containing step information.
    - wandb.Image: A WandB object for logging images.

    This function generates a plot, saves it as a .png image in a folder named after 'plt_key' inside
    the 'plotdir', and displays the plot. It can also log the plot to WandB with optional step information
    if WandB logging is enabled. This function is typically used to visualize and log plots generated
    during training or evaluation.

    Example:
    dump_plot('plot_image', 'training_plot', 'iteration')
    """
    global plotdir, _config

    if not _config.debugging_mode and _config.plot_custom_figs:
        # Create a folder inside the plots folder for each plt_key
        path = osp.join(plotdir, plt_key)
        os.makedirs(path, exist_ok=True)
        path = osp.join(path, filename + '.png')
        plt.savefig(path, dpi=300)
        plt.show()
        step_info = {}
        if step_key is not None:
            step_info[step_key] = _steps[step_key]
        dump_wandb({**{plt_key: wandb.Image(path)}, **step_info})
    else:
        plt.show()


def dump_plot_with_key(plt_key, filename, plt_info=None,
                       plt_kwargs=None, subplot=True,
                       first_col_as_x=False, custom_col_config_list=None,
                       columns=None, step_key=None, keep_plt_key_in_queue_after_plot=False):
    """
    Generate a plot from the data in the plot queue and save it as an image file.

    Parameters:
    - plt_key (str): A key associated with the plot.
    - filename (str): The filename for the saved image.
    - plt_info (dict, optional): Additional plot information.
    - plt_kwargs (dict, optional): Keyword arguments for customizing the plot.
    - subplot (bool, optional): Whether to use subplots for multiple columns of data.
    - first_col_as_x (bool, optional): Whether the first column is used as the x-axis data.
    - custom_col_config_list (list, optional): A list of custom column configurations.
    - columns (list, optional): Column names for the data.
    - step_key (str, optional): A key for tracking the step.
    - keep_plt_key_in_queue_after_plot (bool, optional): Whether to keep the plot key in the queue after plotting.

    Global variables used:
    - _plot_queue: A dictionary containing data for generating plots.
    - _config: A global configuration object containing settings.
    - _plot: A function for generating plots.
    - plotdir: The directory where plot images are saved.
    - _save_csv: A function for saving data to a CSV file.

    This function is used to generate a plot from the data in the plot queue, customize the plot using
    optional arguments, save it as an image file, and optionally save the data to a CSV file.
    It can also control the behavior of subplots, x-axis configuration, column configurations, and more.
    After plotting, it can remove the plot key from the queue if 'keep_plt_key_in_queue_after_plot' is set to False.

    Example:
    dump_plot_with_key('training_plot', 'plot_image', plt_info={'xlabel': 'Timestep'})
    """
    global _plot_queue, _config

    if not _config.plot_custom_figs and not _config.save_custom_figs_data:
        return

    assert plt_key in _plot_queue.keys(), "Plot key is not in plot queue"
    data = np.vstack(_plot_queue[plt_key])
    if _config.plot_custom_figs:
        _ = _plot(data=data,
                  plt_info=plt_info,
                  plt_kwargs=plt_kwargs,
                  subplot=subplot,
                  first_col_as_x=first_col_as_x,
                  custom_col_config_list=custom_col_config_list
                  )
        dump_plot(filename, plt_key=plt_key, step_key=step_key)
    if _config.save_custom_figs_data and not _config.debugging_mode:
        path = osp.join(plotdir, plt_key)
        _save_csv(data=data, path=path, filename=filename, columns=columns)

    # reset _plot_queue corresponding to plt_key
    # _plot_queue[plt_key] = []
    if not keep_plt_key_in_queue_after_plot:
        del _plot_queue[plt_key]


def _plot(data, plt_info, plt_kwargs=None, subplot=True, first_col_as_x=False, custom_col_config_list=None):
    """
        Generate a plot from data with optional customization.

        Parameters:
        - data (numpy.ndarray): The data to be plotted.
        - plt_info (dict): Plot information including labels and styles.
        - plt_kwargs (dict, optional): Customization options for the plot.
        - subplot (bool, optional): Whether to use subplots for multiple columns of data.
        - first_col_as_x (bool, optional): Whether the first column represents the x-axis data.
        - custom_col_config_list (list, optional): List of custom column configurations.

        Returns:
        - plt: The matplotlib plot object.

        Global variables used:
        - _get_color_cycle: A function to get a color cycle for the plot.
        - plt: The matplotlib.pyplot module.

        This function generates a plot from the given data using matplotlib and allows for various customizations.
        It can create subplots, use different y-scales, and more. The function returns the matplotlib plot object
        for further customization or saving.

        Example:
        data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        plt_info = {'ylabel': 'Value', 'xlabel': 'Time'}
        _ = _plot(data, plt_info, plt_kwargs={'linestyle': '--', 'marker': 'o'})
        plt.show()
        """

    # Determine if the first column is for the data in the x-axis
    if first_col_as_x:
        data_x = data[:, 0]
        data_y = data[:, 1:]
    else:
        data_x = np.arange(data.shape[0])
        data_y = data

    if custom_col_config_list is None:
        custom_col_config_list = []
        for item in list(range(data_y.shape[1])):
            custom_col_config_list.append([item])
    # ncols = data_y.shape[1] if custom_col_config_list is None else len(custom_col_config_list)
    ncols = len(custom_col_config_list)

    # Determine yscale: linear, log, etc.
    if "yscale" in plt_info.keys():     # TODO: you may need to accept a list of yscales in the case of multiple plot
        plt.yscale(plt_info["yscale"])

    if ncols == 1:  # If only one column for data
        if plt_kwargs is not None:
            plt.plot(data_x, data_y,  **plt_kwargs)
        else:
            plt.plot(data_x, data_y)
        if "ylabel" in plt_info.keys():
            plt.ylabel(plt_info["ylabel"])
        if "legend" in plt_info.keys():
            if plt_info["legend"] is not None:
                plt.legend(plt_info["legend"], frameon=False)

    else:   # If you have multiple columns of data, use subplot
        colors = _get_color_cycle(data_y.shape[1])      # you need colors to the number of columns you have in data_y
        if subplot:      # you want to add the merged subplot usage only for the subplot case
            f, axs = plt.subplots(ncols, 1) # TODO: you may need to accept other configurations for the subplot
            for col in range(ncols):
                if plt_kwargs is not None:
                    for in_col in custom_col_config_list[col]:
                        axs[col].plot(data_x, data_y[:, in_col], c=next(colors), **plt_kwargs)
                    # plt.plot(data_x, data_y[:, col], c=next(colors), **plt_kwargs)
                else:
                    for in_col in custom_col_config_list[col]:
                        axs[col].plot(data_x, data_y[:, in_col], c=next(colors))
                        # plt.plot(data_x, data_y[:, col], c=next(colors))
                if "ylabel" in plt_info.keys():
                    axs[col].set_ylabel(plt_info["ylabel"][col])

                if "legend" in plt_info.keys():
                    if plt_info["legend"][col] is not None:
                        axs[col].legend(plt_info["legend"][col], frameon=False)
        else:
            for col in range(ncols):
                if plt_kwargs is not None:
                    plt.plot(data_x, data_y[:, col], c=next(colors), **plt_kwargs)
                else:
                    plt.plot(data_x, data_y[:, col], c=next(colors))
            if "legend" in plt_info.keys():
                plt.legend(plt_info["legend"],
                           frameon=False)
            if "ylabel" in plt_info.keys():
                plt.ylabel(plt_info["ylabel"])


    if "xlabel" in plt_info.keys():
        plt.xlabel(plt_info["xlabel"])
    else:
        plt.xlabel("Timestep")

    return plt

def dump_csv(cat_key):
    """
    Dump tabular data to a CSV file.

    Parameters:
    - cat_key (str): The category key indicating the type of data being dumped.

    Global variables used:
    - _tabular_headers: Dictionary containing the headers of tabular data.
    - _tabular_fds: Dictionary of file descriptors for writing tabular data.
    - _tabular_queue: Dictionary containing the tabular data.
    - np: NumPy library for handling numerical data.
    - csv: CSV library for reading and writing CSV files.

    This function writes the tabular data from the specified category to a CSV file. It checks if the header already exists in the CSV file; if not, it writes the header first. If there are additional keys in the data, it updates the header and the CSV file accordingly.

    Example:
    dump_csv('iteration')
    """
    global _tabular_headers
    write_fd = _tabular_fds['w'][cat_key]
    tabular_queue = _tabular_queue[cat_key]
    tabular_queue = [tabular_queue] if not isinstance(tabular_queue, list) else tabular_queue
    for queue in tabular_queue:
        existing_keys = _tabular_headers[cat_key]
        keys = queue.keys()
        if existing_keys is None:  # first time writing on the csv file
            writer = csv.DictWriter(write_fd, fieldnames=list(keys))
            writer.writeheader()
            _tabular_headers[cat_key] = list(keys)
            writer.writerow(queue)
        else:                       # header is already written
            if not set(existing_keys).issuperset(set(keys)):
                joint_keys = set(keys).union(set(existing_keys))
                read_fd = _tabular_fds['r'][cat_key]
                reader = csv.DictReader(read_fd)
                rows = list(reader)
                read_fd.close()
                write_fd.close()
                # make new write_fd
                old_writer_name = write_fd.name
                write_fd = _tabular_fds['w'][cat_key] = open(old_writer_name, 'w')
                writer = csv.DictWriter(write_fd, fieldnames=list(joint_keys))
                writer.writeheader()
                for row in rows:
                    for key in joint_keys:
                        if key not in row:
                            row[key] = np.nan
                writer.writerows(rows)
                _tabular_headers[cat_key] = list(joint_keys)
            else:
                writer = csv.DictWriter(write_fd, fieldnames=_tabular_headers[cat_key])
                for key in _tabular_headers[cat_key]:
                    if key not in queue:
                        queue[key] = np.nan
                writer.writerow(queue)
        write_fd.flush()

def add_stats(key, val):
    """
    Calculate statistics for a specific key in the tabular data.

    Parameters:
    - key (str): The key for which statistics are calculated.
    - val (list or numpy.ndarray): The data associated with the key.

    Global variables used:
    - np: NumPy library for mathematical operations.

    Returns:
    - dict: A dictionary containing the calculated statistics for the specified key,
     including mean, standard deviation, maximum, and minimum values.

    Example:
    stats = add_stats('reward', [1.0, 2.0, 3.0, 4.0, 5.0])
    print(stats)
    {'reward_Mean': 3.0, 'reward_Std': 1.58, 'reward_Max': 5.0, 'reward_Min': 1.0}
    """
    return {
        key + '_Mean': np.mean(val, axis=0),
        key + '_Std': np.std(val, axis=0),
        key + '_Max': np.max(val, axis=0),
        key + '_Min': np.min(val, axis=0),
    }

def clear_tabular_queue(cat_key):
    """
    Clear the tabular data queue for a specific category.

    Parameters:
    - cat_key (str): The category key for which the tabular data queue should be cleared.

    Global variables used:
    - _tabular_queue: A global dictionary that holds tabular data for different categories.

    Example:
    clear_tabular_queue('iteration')
    """
    global _tabular_queue
    _tabular_queue[cat_key] = {}


def _get_color_cycle(n):
    """
        Get an iterator for a color cycle based on the specified colormap.

        Parameters:
        - n (int): The number of colors to generate in the cycle.

        Global variables used:
        - _sns_colormap: A global variable specifying the colormap to use for color cycling.
        - _colormap: A global variable specifying the colormap name.

        Returns:
        - iter: An iterator for cycling through colors based on the specified colormap.

        Example:
        color_cycle = _get_color_cycle(5)
        next(color_cycle)
        """
    global _sns_colormap, _colormap

    # matplotlib colorpallet
    # cm = plt.get_cmap(_colormap)
    # return iter(cm(np.linspace(0, 1, n)))

    return iter(sns.color_palette(_sns_colormap, n))


def notify_completion():
    """
    Notify completion of a task or process via email.

    This function sends an email with the subject "Completed!" to the specified email address. The email body is left empty.

    Global variables used:
    - _config: A global variable for configuration settings.
    - _email: A helper function for sending emails.

    Example:
    notify_completion()
    """
    subject = 'Completed!'
    body = ''
    _email(subject, body)


def notify_failure(error_message):
    """
    Notify a failure or error via email.

    This function sends an email with the subject "Failed" to the specified email address,
    including the provided error message in the email body.

    Parameters:
    - error_message (str): The error message to be included in the email body.

    Global variables used:
    - _config: A global variable for configuration settings.
    - _email: A helper function for sending emails.

    Example:
    error_message = "An error occurred during the execution."
    notify_failure(error_message)
    """
    subject = 'Failed'
    body = str(error_message)
    _email(subject, body)


def _email(subject, body):
    """
    Send an email with the specified subject and body.

    This function is used to send an email with the given subject and body to the email
    address specified in the global variables.

    Parameters:
    - subject (str): The subject of the email.
    - body (str): The content or body of the email.

    Global variables used:
    - _config: A global variable for configuration settings.
    - os: The Python os module for environment variable access.
    - smtplib: The Python smtplib module for sending emails.
    - EmailMessage: A class from the email.message module for creating email messages.

    Example:
    subject = "Test Email"
    body = "This is a test email message."
    _email(subject, body)
    """
    if _config.debugging_mode:
        return

    EMAIL_ADDRESS = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')
    EMAIL_APPLICATION_PASSWORD = os.environ.get('G_APP_PASS')
    if not EMAIL_ADDRESS or not (EMAIL_PASSWORD or EMAIL_APPLICATION_PASSWORD):
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS
    msg.set_content(body)

    # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    #     smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    #     smtp.send_message(msg)

    try:
        # Attempt to connect and send the email using the application-specific password
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_APPLICATION_PASSWORD)
            smtp.send_message(msg)
            print("Email sent using application-specific password.")
    except smtplib.SMTPAuthenticationError:
        # If the application-specific password fails, try with the regular password
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("Email sent using regular password.")

def _save_csv(data, path, filename, columns=None, index=False):
    """
        Save data as a CSV file.

        This function takes the data, path, filename, and optional column information and saves the data as a CSV file.

        Parameters:
        - data (ndarray): The data to be saved as a CSV file.
        - path (str): The directory path where the CSV file will be saved.
        - filename (str): The name of the CSV file (without the file extension).
        - columns (list, optional): A list of column names for the CSV file. Default is None.
        - index (bool, optional): Whether to include the index in the CSV file. Default is False.

        Global variables used:
        - osp: The Python os.path module for handling file paths.
        - DataFrame: A class from the pandas module for creating data frames.

        Example:
        data = np.array([[1, 2], [3, 4]])
        path = "/data"
        filename = "example_data"
        columns = ["Column1", "Column2"]
        _save_csv(data, path, filename, columns, index=False)
        """
    file = osp.join(path, filename + '.csv')
    data_frame = DataFrame(data, columns=columns)
    data_frame.to_csv(file, index=index, header=False if columns is None else True)

def dump_dict2json(dictionary, filename):
    """
    Dump a dictionary as a JSON file.

    This function takes a dictionary and a filename and saves the dictionary as a JSON file with the specified filename.

    Parameters:
    - dictionary (dict): The dictionary to be saved as a JSON file.
    - filename (str): The name of the JSON file (without the file extension).

    Global variables used:
    - logdir: The directory where the JSON file will be saved.
    - _config: A global configuration object.

    Example:
    data = {"name": "John", "age": 30}
    filename = "example_data"
    dump_dict2json(data, filename)
    """
    global logdir, _config
    if not _config.debugging_mode:
        file = osp.join(logdir, filename + '.json')
        with open(file, "w") as f:
            json.dump(dictionary, f, indent=4)
        log(f'{filename}.json saved')

def get_plot_queue_by_key(plt_key):
    """
    Get the plot queue data for a specific plot key.

    This function retrieves the data stored in the plot queue for a specific plot key.
    The plot queue is a global variable used to store data for generating custom plots.

    Parameters:
    - plt_key (str): The key associated with the plot data in the plot queue.

    Global variables used:
    - _plot_queue: The plot queue containing data for custom plots.

    Returns:
    - list: A list of data arrays associated with the specified plot key.

    Example:
    data = get_plot_queue_by_key("custom_plot")
    print(data)
    [array([1, 2, 3]), array([4, 5, 6])]
    """
    global _plot_queue
    assert plt_key in _plot_queue.keys(), "Plot key is not in plot queue"
    return _plot_queue[plt_key]


def set_plot_queue_by_key(plt_key, data):
    """
    Set the plot queue data for a specific plot key.

    This function allows you to set or update the data in the plot queue for a specific
    plot key. The plot queue is a
    global variable used to store data for generating custom plots.

    Parameters:
    - plt_key (str): The key associated with the plot data in the plot queue.
    - data (list): A list of data arrays to be associated with the specified plot key.

    Global variables used:
    - _plot_queue: The plot queue containing data for custom plots.

    Example:
    data = [array([1, 2, 3]), array([4, 5, 6])]
    set_plot_queue_by_key("custom_plot", data)
    """
    global _plot_queue
    _plot_queue[plt_key] = data