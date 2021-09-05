from copy import deepcopy
import torch
import datetime
import os.path as osp
import os, glob
import numpy as np
from attrdict import AttrDict
from scipy.signal import lfilter
import pickle
import importlib
import shutil
from utils.seed import rng


def hard_copy(source, requires_grad=False):
    copied = deepcopy(source)
    # set requires_grad
    for p in copied.parameters():
        p.requires_grad = requires_grad
    return copied


@torch.no_grad()
def polyak_update(target, source, tau):
    for p_targ, p in zip(target.parameters(), source.parameters()):
        p_targ.data.mul_(1 - tau)
        p_targ.data.add_(tau * p.data)

def freeze_net(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    for p in net.parameters():
        p.requires_grad = True


def torchify(x, dtype=torch.float32, device='cpu', requires_grad=False):
    if torch.is_tensor(x):
        x.requires_grad_(requires_grad)
        x.to(dtype=dtype)
        x.to(device='cuda' if device == 'gpu' else 'cpu')
        return x
    if isinstance(x, dict): # dict of torch.tensors
        out = {k: torchify(v, dtype, device, requires_grad) for k, v in x.items()}
        return out
    if isinstance(x, np.ndarray) and not(x.shape[0] == 0) and isinstance(x[0], dict):
        out = np.hstack([torchify(dic) for dic in x])
        return out
    out = torch.as_tensor(x, dtype=dtype, device=torch.device('cuda' if device == 'gpu' else 'cpu'))
    out.requires_grad_(requires_grad)
    return out


def to_device(x, device):
    # TODO: check if it is no-ops when the device is the same
    if device == 'gpu':
        return x.cuda()
    if device == 'cpu':
        return x.cpu()


def get_ac_space_info(ac_space):
    """returns action space shape"""
    # TODO add multidiscrete option here
    from gym.spaces import Box
    if isinstance(ac_space, Box):  # Continuous
        is_discrete = False
        return ac_space.shape[0], is_discrete
    else:  # Discrete
        is_discrete = True
        return ac_space.n, is_discrete


def get_save_checkpoint_name(run_root_dir, episode):
    filename = 'ckpt_%s_ep%d.pt' % (get_timestamp(for_logging=False), episode)
    path = osp.join(run_root_dir, 'checkpoints')
    os.makedirs(path, exist_ok=True)
    return osp.join(path, filename)

def get_load_checkpoint_name(current_root, load_run_name, timestamp):
    # change run name in the current run name to load_run_name, keep everything else the same
    path_split = current_root.split(os.sep)
    path_split[-2] = load_run_name
    path = osp.join(os.sep, *path_split, 'checkpoints')
    timestamp = get_last_timestamp(path) if timestamp == 'last' else timestamp
    filename = 'ckpt_%s.pt' % timestamp
    return osp.join(path, filename)

def get_loaded_config(project_dir, load_run_name):
    path = osp.join(project_dir, 'wandb', load_run_name, 'files')
    return load_config_from_py(path=path)

def get_last_timestamp(ckpt_dir):
    os.chdir(ckpt_dir)
    ckpt_list = glob.glob("*.pt")
    last_ckpt = ckpt_list[-1]
    last_ckpt_timestamp = last_ckpt[5:-3]  # checkpoint names ars in the format ckpt_YYYYMMDD_HHMMSS.pt
    return last_ckpt_timestamp

def get_timestamp(for_logging=True):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S') if for_logging else now.strftime('%Y%m%d_%H%M%S')
    timestamp_prefix = "%s | " % timestamp if for_logging else timestamp
    return timestamp_prefix

def n_last_eval_video_callable(n, value):
    def video_callable(x):
        return x % value in range(value-n, value)
    return video_callable

#
# def namedtuple2dict(tup):
#     """Converts list of namedtuples to dictionary of numpy arrays"""
#     d = DotMap(tup[0]._asdict())      # _asdict returns ordered dict
#     for i in range(1, len(tup)):
#       for k in d.keys():
#         d[k] = np.concatenate((d[k], tup[i]._asdict()[k]), axis=0)
#     return d


def namedtuple2dict(tup):
    """Converts list of namedtuples to dictionary of numpy arrays
    input is list of namedtuples. See Safe RL implementation Notes for input output specificaiton
    For 4D input:
        dimension order: ensemble, num_sequence, horizon, data shape
    For 3D input:
        dimension order: num_sequence, horizon, data shape

    """


    keys = tup[0]._asdict().keys()
    d = AttrDict({key: [] for key in keys})
    for i in range(len(tup)):
        for k in d.keys():
            d[k].append(tup[i]._asdict()[k])
    for k in d.keys():
        d[k] = np.stack(d[k], axis=d[k][0].ndim - 1)
    return d


def scaler(x, lim_from, lim_to):
    x_init_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x_dim = x.shape[1:]     # assumes the first dimension is the batch dim

    def boundconverter(a):
        if isinstance(a, float):
            a = np.repeat(a, np.prod(x_dim))
        if isinstance(a, np.ndarray):
            a = a.reshape(1, *x_dim)
        return a
    low_from, high_from = lim_from
    low_to, high_to = lim_to

    low_from = boundconverter(low_from)
    high_from = boundconverter(high_from)
    low_to = boundconverter(low_to)
    high_to = boundconverter(high_to)
    out = low_to + (high_to - low_to) * (x - low_from) / (high_from - low_from)
    return out.reshape(x_init_shape)


def discount_cumsum(x, discount, return_first=False):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        array x, [[x0_1, x1_1, x2_1],
                [x0_2, x1_2, x2_2]]
    output: [[x0_1 + discount * x1_1 + discount^2 * x2_1, x1_1 + discount * x2_1, x2_1],
            [x0_2 + discount * x1_2 + discount^2 * x2_2, x1_2 + discount * x2_2, x2_2]]

    """
    if x.ndim > 2:
        x = x.squeeze(axis=-1)

    cumsum = lfilter([1], [1, float(-discount)], x[..., ::-1], axis=-1)[..., ::-1]
    if return_first:
        to_return = cumsum[..., 0][..., np.newaxis]
        return to_return
    return cumsum[..., np.newaxis]


def train_valid_split(data_size, holdout_ratio):
    num_valid = int(holdout_ratio * data_size)
    num_train = data_size - num_valid

    idx_perm = rng.permutation(data_size)
    train_ids = idx_perm[:num_train]
    valid_ids = idx_perm[num_train:]
    return train_ids, valid_ids


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def add_noise(data_inp, noise_to_signal=0.01):
    data = deepcopy(data_inp)  # (num data points, dim)

    # mean of data
    mean_data = np.mean(data, axis=0)

    # if mean is 0,
    # make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    # width of normal distribution to sample noise from
    # larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noise_to_signal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + rng.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data


def dump_pickle(path, filename, obj):
    filename = filename + '.pkl'
    file = osp.join(path, filename)
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path, filename):
    file = osp.join(path, filename)
    with open(file, 'rb') as f:
        return pickle.load(f)


def load_config_from_py(path):
    module_name = path.replace('/', '.')
    module_name += '.config'
    mod = importlib.import_module(module_name)
    config_cls = getattr(mod, 'Config')
    return config_cls()


def save_config_as_py(save_path):
    dst = os.path.join(save_path, 'config.py')
    shutil.copyfile(src='config.py', dst=dst)


def np_object2dict(x, ret_AttrDict=True):
    keys = x[0].keys()
    out = {k: [] for k in keys}
    num_items = x.shape[0] if isinstance(x, np.ndarray) else len(x)
    for i in range(num_items):
        for k in keys:
            out[k].append(x[i][k])
    for k in keys:
        out[k] = torch.vstack(out[k]) if torch.is_tensor(out[k][0]) else np.vstack(out[k])
    return AttrDict(out) if ret_AttrDict else out


def e_and(*args):   # element-wise and
    if not isinstance(args[0], list):   # if inputs are single elements
        return all(args)
    return [all(tup) for tup in zip(*args)]


def e_or(*args):    # element-wise or
    if not isinstance(args[0], list):   # if inputs are single elements
        return any(args)
    return [any(tup) for tup in zip(*args)]


def e_not(arg):     # element-wise not
    if isinstance(arg, list):
        return list(np.invert(np.array(arg)))
    return not arg


def theta2vec(theta, add_z_axis=False):
    if add_z_axis:
        return np.array([np.cos(theta), np.sin(theta), 0.0])
    return np.array([np.cos(theta), np.sin(theta)])


# TODO : Fix this
def obs_tensor2np(func):
    def wrapper_tensor2np(obs):
        if torch.is_tensor(obs):
            obs = obs.numpy()
            return func(obs)
    return wrapper_tensor2np

def isvec(x):
    assert isinstance(x, np.ndarray) or torch.is_tensor(x)
    return x.ndim == 1

def np_isvectorable(x):
    return x.squeeze().ndim == 1

def apply_mask_to_dict_of_arrays(dict_of_arrays, mask):
    out = {}
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask, dtype=np.bool_)
    for k in dict_of_arrays.keys():
        out[k] = dict_of_arrays[k][mask, ...]
    return AttrDict(out)
