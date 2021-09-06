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

mpl.rcParams['agg.path.chunksize'] = 10000
mpl.rcParams['text.usetex'] = True

_sns_colormap = 'husl' # other options, husl, rocket, Paired, tab10
_colormap = 'jet'
_config = None

# strings
_root_dir = ''
logdir = ''
plotdir = ''

# dicts
_tabular_fds = {}
_tabular_headers = {}
_tabular_queue = {}
_steps = {}
_plot_queue = {}

# bool
_plotter_in_eval_mode = False


def initialize(config, root_dir):
    global _config

    _config = config
    # initialize wandb
    if not _config.debugging_mode:
        _initialize_wandb(root_dir)
    # setup csv and tabular
    _initialize_tabular()
    # create plot dir
    _create_plot_dir()

def _initialize_wandb(root_dir):
    global logdir

    wandb_dir = osp.join(root_dir, _config.results_dir, _config.wandb_project_name)
    os.makedirs(wandb_dir, exist_ok=True)
    if _config.resume:
        resume = "must"
        id = _config.load_run_id
    else:
        resume = "never"
        id = wandb.util.generate_id()

    wandb.login()
    wandb.init(id=id,
               config=_config,
               project=_config.wandb_project_name,
               dir=wandb_dir,
               resume=resume)
    logdir = wandb.run.dir

def _initialize_tabular():
    global _tabular_fds, _tabular_headers, _tabular_queue, _steps
    csv_file_names = ['episode', 'evaluation_episode', 'iteration', 'epoch', 'cbf_epoch', 'dynamics_epoch']
    tabular_dir = osp.join(logdir, 'tabular')
    if not _config.debugging_mode:
        os.makedirs(tabular_dir, exist_ok=True)
        csv_dir = [osp.join(tabular_dir, file + '_progress.csv') for file in csv_file_names]
        _tabular_fds['w'] = {k: open(file, mode='w') for k, file in zip(csv_file_names, csv_dir)}
        _tabular_fds['r'] = {k: open(file, mode='r') for k, file in zip(csv_file_names, csv_dir)}
    _tabular_headers = {k: None for k in csv_file_names}
    # make empty queue for each csv files
    _tabular_queue = {k: {} for k in csv_file_names}
    _steps = {k: 0 for k in csv_file_names}


def _create_plot_dir():
    global plotdir
    plotdir = osp.join(logdir, 'plots')
    os.makedirs(plotdir, exist_ok=True)


def log(msg, color='green'):
    """Print a colorized message to stdout."""
    if _config.add_timestamp:
        msg = get_timestamp() + msg
    print(colorize(msg, color, bold=True))


def push_tabular(data, cat_key):
    global _tabular_queue
    """use this method when you have a list of dictionary to be added to _tabular_queue for later use in dump_csv"""
    _tabular_queue[cat_key] = data


def add_tabular(data_dict, cat_key='iteration', stats=False, for_csv=False):
    if stats:
        for k, v in data_dict.items():
            stats = add_stats(k, v)
            _log_tabular(stats, cat_key)
    else:
        _log_tabular(data_dict, cat_key)


def _log_tabular(data_dict, cat_key):
    global _tabular_queue

    for k, v in data_dict.items():
        _tabular_queue[cat_key][k] = v


def dump_tabular(cat_key='iteration', log=False, wandb_log=False, csv_log=False):
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
    wandb.log(queue)


def push_plot(data, plt_key, row_append=False):
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
    global _plotter_in_eval_mode
    _plotter_in_eval_mode = True


def set_plotter_in_train_mode():
    global _plotter_in_eval_mode
    _plotter_in_eval_mode = False


def dump_plot(filename, plt_key):
    global plotdir, _config

    if not _config.debugging_mode and _config.plot_custom_figs:
        # Create a folder inside the plots folder for each plt_key
        path = osp.join(plotdir, plt_key)
        os.makedirs(path, exist_ok=True)
        path = osp.join(path, filename + '.png')
        plt.savefig(path, dpi=300)
        plt.show()
        dump_wandb({plt_key: wandb.Image(path)})
    else:
        plt.show()


def dump_plot_with_key(plt_key, filename, plt_info=None,
                       plt_kwargs=None, subplot=True,
                       first_col_as_x=False, custom_col_config_list=None,
                       columns=None):
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
        dump_plot(filename, plt_key)
    if _config.save_custom_figs_data:
        path = osp.join(plotdir, plt_key)
        _save_csv(data=data, path=path, filename=filename, columns=columns)

    # reset _plot_queue corresponding to plt_key
    # _plot_queue[plt_key] = []
    del _plot_queue[plt_key]


def _plot(data, plt_info, plt_kwargs=None, subplot=True, first_col_as_x=False, custom_col_config_list=None):

    # determine if the first column is for the data in the x-axis
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

    # determine yscale: linear, log, etc.
    if "yscale" in plt_info.keys():     # TODO: you may need to accept a list of yscales in the case of multiple plot
        plt.yscale(plt_info["yscale"])

    if ncols == 1:  # if only one column for data
        if plt_kwargs is not None:
            plt.plot(data_x, data_y,  **plt_kwargs)
        else:
            plt.plot(data_x, data_y)
        if "ylabel" in plt_info.keys():
            plt.ylabel(plt_info["ylabel"])
    else:   # if you have multiple columns of data, use subplot
        colors = _get_color_cycle(data_y.shape[1])      # you need colors to the number of columns you have in data_y
        if subplot:      # you wanna add the merged subplot usage only for the subplot case
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
    return {
        key + '_Mean': np.mean(val, axis=0),
        key + '_Std': np.std(val, axis=0),
        key + '_Max': np.max(val, axis=0),
        key + '_Min': np.min(val, axis=0),
    }

def clear_tabular_queue(cat_key):
    global _tabular_queue
    _tabular_queue[cat_key] = {}


def _get_color_cycle(n):
    global _sns_colormap, _colormap

    # matplotlib colorpallet
    # cm = plt.get_cmap(_colormap)
    # return iter(cm(np.linspace(0, 1, n)))

    return iter(sns.color_palette(_sns_colormap, n))


def notify_completion():
    subject = 'Completed!'
    body = ''
    _email(subject, body)


def notify_failure(error_message):
    subject = 'Failed'
    body = str(error_message)
    _email(subject, body)


def _email(subject, body):
    if _config.debugging_mode:
        return

    EMAIL_ADDRESS = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS
    msg.set_content(body)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

def _save_csv(data, path, filename, columns=None, index=False):
    file = osp.join(path, filename + '.csv')
    data_frame = DataFrame(data, columns=columns)
    data_frame.to_csv(file, index=index, header=False if columns is None else True)

def dump_dict2json(dictionary, filename):
    global logdir, _config
    if not _config.debugging_mode:
        file = osp.join(logdir, filename + '.json')
        with open(file, "w") as f:
            json.dump(dictionary, f, indent=4)
        log(f'{filename}.json saved')
