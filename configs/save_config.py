from config import Config
from utils.misc import get_timestamp, dump_pickle
import os
import shutil

def save_config(alg, env):
    """run this method to save current config"""
    # specify path
    path = os.path.dirname(os.path.realpath(__file__))

    # make file name
    time_stamp = get_timestamp(for_logging=False)
    filename = '%s_%s_%s' % (alg, env, time_stamp)

    # instantiate config file
    config = Config()

    # dump pickle file
    dump_pickle(path=path, filename=filename, obj=config)

    # save config as .py file
    config_file = os.path.join(path, '..', 'config.py')
    filename += '.py'
    shutil.copyfile(config_file, filename)

if __name__ == '__main__':
    alg = 'ddpg'
    env = 'pendulum'
    save_config(alg, env)


