import torch
from torch.autograd import grad
from torch.autograd.functional import jacobian
import numpy as np

def get_jacobian(net, x, batch_dim_first=False, squeeze_on_one_output=True, create_graph=False):
    # x shape: batch_size x num_inputs
    # net input shape: batch_size x num_inputs
    # net output shape: batch_size x num_inputs
    jac = jacobian(net, x, create_graph=create_graph).sum(dim=0)
    if jac.size(0) == 1 and squeeze_on_one_output:      # output dim is 0
        return jac.squeeze(dim=0)
    if batch_dim_first:
        # reshape the jacobian: batch_size, num_inputs, num_outputs
        return torch.moveaxis(jac, 0, -1)
    # reshape the jacobian: num_outputs, batch_size, num_inputs
    return jac

def get_vjp(net, x, v, numpy_output=False):
    # computes jacobian (dot) v for (numpy or tensor inputs x and v)
    x = x if x.is_tensor() else torch.tensor(x)
    jac = get_jacobian(net, x)
    if isinstance(v, np.ndarray) and not numpy_output:
        v = torch.tensor(v, dtype=torch.float32)
    if numpy_output:
        v = v if isinstance(v, np.ndarray) else v.numpy()
        return np.dot(jac.detach().numpy(), v)
    return torch.dot(jac(net, x), v)


def Rop(y, x, v):
    """Computes an Rop.

    Arguments:
      y (Variable): output of differentiated function
      x (Variable): differentiated input
      v (Variable): vector to be multiplied with Jacobian from the right
    """
    w = torch.ones_like(y, requires_grad=True)
    return torch.autograd.grad(torch.autograd.grad(y, x, w), w, v)

# def get_grad(y, x):
#     w = torch.ones_like(y, requires_grad=True)
#     return torch.autograd.grad(y, x, w)


def get_grad(net, x,
             create_graph=True, retain_graph=True, allow_unused=True,
             squeeze_on_one_output=True, batch_dim_first=False):
    x.requires_grad_()
    y = net(x)
    num_out = y.size(1)
    jac = []
    for dim in range(num_out):
        w = torch.zeros_like(y)
        w[:, dim] += 1.0
        grad_x, = torch.autograd.grad(y, x, w, retain_graph=retain_graph, create_graph=create_graph, allow_unused=allow_unused)
        jac.append(grad_x)
    jac = torch.stack(jac)
    if jac.size(0) == 1 and squeeze_on_one_output:      # output dim is 0
        return jac.squeeze(dim=0)

    if batch_dim_first:
        # reshape the jacobian: batch_size, num_inputs, num_outputs
        return torch.moveaxis(jac, 0, -1)
    return jac
