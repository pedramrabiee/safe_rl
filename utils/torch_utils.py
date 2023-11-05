import torch
from attrdict import AttrDict
import numpy as np


def row_wise_dot(a, b):
    if torch.is_tensor(a):
        return (a * b).sum(dim=-1).unsqueeze(dim=-1)
    if isinstance(a, np.ndarray):
        return np.expand_dims((a * b).sum(axis=-1), axis=-1)


def apply_mask_to_dict_of_tensors(dict_of_tensors, mask):
    out = {}
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, dtype=torch.bool)
    for k in dict_of_tensors.keys():
        out[k] = dict_of_tensors[k][mask, ...]
    return AttrDict(out)

def get_tensor_blueprint(x):
    assert torch.is_tensor(x), 'The input to get_tensor_blueprint needs to be a torch tensor'
    return dict(dtype=x.dtype,
                device=x.device.type,
                requires_grad=x.requires_grad)

# SAVING AND LOADING TOOLS
# def map_state_dict_on_dict(my_dict):
#     return {k: v.state_dict() for k, v in my_dict.items()}
#
#
# def itermap_state_dict(x):
#     out = {}
#     if not isinstance(x[next(iter(x))], dict):
#         return map_state_dict_on_dict(x)
#     for k in x.keys():
#         out[k] = itermap_state_dict(x[k])
#     return out


# def map_load_state_dict_on_dict(state_dict, load_to):
#     return {k: v.load_state_dict(state_dict[k]) for k, v in load_to.items()}


def load_state_dict_from_dict(state_dict, load_to):
    func = lambda to, arg: to.load_state_dict(arg)
    iterapply_on_dict(apply_to=load_to,
                      func=func,
                      arg=state_dict)

#

# def map_detach_on_dict(my_dict):
#     return {k: v.detach() for k, v in my_dict.items()}
#
#
# def itermap_detach(x):
#     out = {}
#     if not isinstance(x[next(iter(x))], dict):
#         return map_detach_on_dict(x)
#     for k in x.keys():
#         out[k] = itermap_detach(x[k])
#     return out


def map_on_dict(x, func):
    return {k: func(v) for k, v in x.items()}


def itermap_on_dict(x, func):
    out = {}
    if not isinstance(x[next(iter(x))], dict):
        return map_on_dict(x, func)
    for k in x.keys():
        if x[k]:
            out[k] = itermap_on_dict(x[k], func)
    return out


def apply_on_dict(apply_to, func, arg):
    return {k: func(v, arg[k]) for k, v in apply_to.items()}


def iterapply_on_dict(apply_to, func, arg):
    if not isinstance(apply_to[next(iter(apply_to))], dict):
        apply_on_dict(apply_to=apply_to, func=func, arg=arg)
        return
    for k in apply_to.keys():
        iterapply_on_dict(apply_to=apply_to[k], func=func, arg=arg[k])


def softmin(x, rho, conservative=False):
    return softmax(x=x, rho=-rho, conservative=conservative)


def softmax(x, rho, conservative=True):
    res = 1 / rho * torch.logsumexp(rho * x, dim=0)
    return res - np.log(x.size(0))/rho if conservative else res