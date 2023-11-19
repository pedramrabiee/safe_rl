import torch

from utils.misc import scaler, scaler_torch

# list
ac_old_bounds = []
ac_new_bounds = []
ac_old_bounds_torch = []
ac_new_bounds_torch = []
# obs_old_bounds = []
# obs_new_bounds = []


def initialize(bounds):
    global ac_old_bounds, ac_new_bounds, ac_old_bounds_torch, ac_new_bounds_torch
    # global obs_old_bounds, obs_new_bounds
    ac_old_bounds = [bounds.old.low, bounds.old.high]
    ac_new_bounds = [bounds.new.low, bounds.new.high]
    ac_old_bounds_torch = [torch.as_tensor(arr) for arr in ac_old_bounds]
    ac_new_bounds_torch = [torch.as_tensor(arr) for arr in ac_new_bounds]

    # obs_old_bounds = [bounds.old.obs.low, bounds.old.obs.high]
    # obs_new_bounds = [bounds.new.obs.low, bounds.new.obs.high]


def action2newbounds(ac):
    global ac_old_bounds, ac_new_bounds, ac_old_bounds_torch, ac_new_bounds_torch
    if torch.is_tensor(ac):
        return scaler_torch(x=ac, lim_from=ac_old_bounds_torch, lim_to=ac_new_bounds_torch)
    return scaler(x=ac, lim_from=ac_old_bounds, lim_to=ac_new_bounds)


def action2oldbounds(ac):
    global ac_old_bounds, ac_new_bounds, ac_old_bounds_torch, ac_new_bounds_torch
    if torch.is_tensor(ac):
        return scaler_torch(x=ac, lim_from=ac_new_bounds_torch, lim_to=ac_old_bounds_torch)
    return scaler(x=ac, lim_from=ac_new_bounds, lim_to=ac_old_bounds)


# def obs2newbounds(obs):
#     global obs_old_bounds, obs_new_bounds
#     return scaler(x=obs, lim_from=obs_old_bounds, lim_to=obs_new_bounds)
#
#
# def obs2oldbounds(obs):
#     global obs_old_bounds, obs_new_bounds
#     return scaler(x=obs, lim_from=obs_new_bounds, lim_to=obs_old_bounds)
