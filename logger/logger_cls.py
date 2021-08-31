import wandb
import numpy as np
import os.path as osp
import os
from utils.console import colorize
from utils.misc import get_timestamp
import csv

# This version is outdated. Refer to logger.py for the newest version
class Logger:
    """This is the class that handles logging."""

    def __init__(self, config, root_dir):
        self._config = config

        #initialize wandb
        self._initialize_wandb(root_dir)

        # setup csv and tabular
        self._initialize_tabular()

    def _initialize_wandb(self, root_dir):
        wandb_dir = osp.join(root_dir, self._config.results_dir, self._config.wandb_project_name)
        os.makedirs(wandb_dir, exist_ok=True)
        if self._config.resume:
            resume = "must"
            id = self._config.load_run_id
        else:
            resume = "never"
            id = wandb.util.generate_id()

        wandb.login()
        wandb.init(id=id,
                   config=self._config,
                   project=self._config.wandb_project_name,
                   dir=wandb_dir,
                   resume=resume)

        self.logdir = wandb.run.dir

    def _initialize_tabular(self):
        tabular_dir = osp.join(self.logdir, 'tabular')
        os.makedirs(tabular_dir, exist_ok=True)
        csv_file_names = ['episode', 'iteration', 'evaluation', 'epoch']
        csv_dir = [osp.join(tabular_dir, file + '_progress.csv') for file in csv_file_names]
        self._tabular_fds = {}
        self._tabular_fds['w'] = {k: open(file, mode='w') for k, file in zip(csv_file_names, csv_dir)}
        self._tabular_fds['r'] = {k: open(file, mode='r') for k, file in zip(csv_file_names, csv_dir)}
        self._tabular_headers = {k: None for k in csv_file_names}
        # make empty queue for each csv files
        self._tabular_queue = {k: {} for k in csv_file_names}
        self._steps = {k: 0 for k in csv_file_names}

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        if self._config.add_timestamp:
            msg = get_timestamp() + msg
        print(colorize(msg, color, bold=True))

    def push_tabular(self, data, cat_key):
        """use this method when you have a list of dictionary to be added to _tabular_queue for later use in dump_csv"""
        self._tabular_queue[cat_key] = data

    def add_tabular(self, data_dict, cat_key='iteration', stats=False, for_csv=False):
        if stats:
            for k, v in data_dict.items():
                stats = self.add_stats(k, v)
                self._log_tabular(stats, cat_key)
        else:
            self._log_tabular(data_dict, cat_key)

    def _log_tabular(self, data_dict, cat_key):
        for k, v in data_dict.items():
            self._tabular_queue[cat_key][k] = v

    def dump_tabular(self, cat_key='iteration', log=False, wandb_log=False, csv_log=False):
        queue = self._tabular_queue[cat_key]
        self._steps[cat_key] += 1
        queue[cat_key] = self._steps[cat_key]
        if log:
            self.dump_console(queue)
        if wandb_log and self._config.use_wandb:
            self.dump_wandb(queue)
        if csv_log:
            self.dump_csv(cat_key)
        self.clear_tabular_queue(cat_key)

    @staticmethod
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

    @staticmethod
    def dump_wandb(queue):
        wandb.log(queue)

    def dump_csv(self, cat_key):
        write_fd = self._tabular_fds['w'][cat_key]
        tabular_queue = self._tabular_queue[cat_key]
        tabular_queue = [tabular_queue] if not isinstance(tabular_queue, list) else tabular_queue
        for queue in tabular_queue:
            existing_keys = self._tabular_headers[cat_key]
            keys = queue.keys()
            if existing_keys is None:  # first time writing on the csv file
                writer = csv.DictWriter(write_fd, fieldnames=list(keys))
                writer.writeheader()
                self._tabular_headers[cat_key] = list(keys)
                writer.writerow(queue)
            else:                       # header is already written
                if not set(existing_keys).issuperset(set(keys)):
                    joint_keys = set(keys).union(set(existing_keys))
                    read_fd = self._tabular_fds['r'][cat_key]
                    reader = csv.DictReader(read_fd)
                    rows = list(reader)
                    read_fd.close()
                    write_fd.close()
                    # make new write_fd
                    old_writer_name = write_fd.name
                    write_fd = self._tabular_fds['w'][cat_key] = open(old_writer_name, 'w')
                    writer = csv.DictWriter(write_fd, fieldnames=list(joint_keys))
                    writer.writeheader()
                    for row in rows:
                        for key in joint_keys:
                            if key not in row:
                                row[key] = np.nan
                    writer.writerows(rows)
                    self._tabular_headers[cat_key] = list(joint_keys)
                else:
                    writer = csv.DictWriter(write_fd, fieldnames=self._tabular_headers[cat_key])
                    for key in self._tabular_headers[cat_key]:
                        if key not in queue:
                            queue[key] = np.nan
                    writer.writerow(queue)
            write_fd.flush()

    @staticmethod
    def add_stats(key, val):
        return {
            key + '_Mean': np.mean(val, axis=0),
            key + '_Std': np.std(val, axis=0),
            key + '_Max': np.max(val, axis=0),
            key + '_Min': np.min(val, axis=0),
        }

    def clear_tabular_queue(self, cat_key):
        self._tabular_queue[cat_key] = {}

