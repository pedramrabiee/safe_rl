import torch
from torch.autograd import grad
from torch.autograd.functional import jacobian
import numpy as np


def get_grad(func, x, **kwargs):
    """
    Compute the gradient of a given function with respect to its input.

    This function calculates the gradient of the function 'func' with respect to the input 'x'. It temporarily
    sets 'x' to require gradients, computes the gradient, and then restores the original gradient requirement.

    Args:
        func (callable): The function for which the gradient is computed.
        x (torch.Tensor): The input data with respect to which differentiation is performed.
        **kwargs: Additional keyword arguments passed to torch.autograd.grad.

    Returns:
        torch.Tensor: The gradient of 'func' with respect to 'x'.
    """
    requires_grad = x.requires_grad
    x.requires_grad_()
    output = grad(func(x), x, **kwargs)
    x.requires_grad_(requires_grad=requires_grad)
    return output[0]

def get_value_and_grad(func, x, **kwargs):
    """
    Compute the gradient of a given function with respect to its input and return the function's value.

    This function calculates the gradient of the function 'func' with respect to the input 'x' and also returns
    the value of the function 'func(x)'. It temporarily sets 'x' to require gradients, computes the gradient,
    and then restores the original gradient requirement.

    Args:
        func (callable): The function for which the gradient is computed.
        x (torch.Tensor): The input data with respect to which differentiation is performed.
        **kwargs: Additional keyword arguments passed to torch.autograd.grad.

    Returns:
        (torch.Tensor, torch.Tensor): Tuple containing the gradient of 'func' with respect to 'x' and the value of 'func(x)'.
    """
    requires_grad = x.requires_grad
    x.requires_grad_()
    output = func(x)  # Calculate func(x) before computing the gradient
    gradient = grad(output, x, **kwargs)
    x.requires_grad_(requires_grad=requires_grad)
    return output, gradient[0]


def get_jacobian_from_net(net, x, batch_dim_first=False, squeeze_on_one_output=True, create_graph=False):
    """
    Compute the Jacobian matrix of a neural network 'net' with respect to its input 'x'.

    Args:
        net (torch.nn.Module): The neural network for which the Jacobian is computed.
        x (torch.Tensor): The input data with shape (batch_size x num_inputs).
        batch_dim_first (bool, optional): If True, reshape the Jacobian as (batch_size, num_inputs, num_outputs).
            If False (default), reshape as (num_outputs, batch_size, num_inputs).
        squeeze_on_one_output (bool, optional): If True and the output dimension of the Jacobian is 1,
            squeeze the resulting Jacobian to (batch_size x num_inputs) if 'batch_dim_first' is False,
            or to (num_inputs x batch_size) if 'batch_dim_first' is True.
        create_graph (bool, optional): Whether to create a graph for higher-order derivatives.

    Returns:
        torch.Tensor: The Jacobian matrix with shape (batch_size x num_outputs x num_inputs) if 'batch_dim_first' is True,
        or (num_outputs x batch_size x num_inputs) if 'batch_dim_first' is False.
        If the output dimension is 1 and 'squeeze_on_one_output' is True, the shape may be adjusted accordingly.
    """

    # FIXME: jacobian probably doesn't work with nn.Module object. Check this and fix
    # Compute the Jacobian
    jac = jacobian(net, x, create_graph=create_graph).sum(dim=0)

    if jac.size(0) == 1 and squeeze_on_one_output:
        # Squeeze the Jacobian if output dimension is 1
        return jac.squeeze(dim=0)

    if batch_dim_first:
        # Reshape the Jacobian as (batch_size, num_inputs, num_outputs)
        return torch.moveaxis(jac, 0, -1)

    # Reshape the Jacobian as (num_outputs, batch_size, num_inputs)
    return jac


def get_grad_from_net(net, x, create_graph=True, retain_graph=True, allow_unused=True,
                      squeeze_on_one_output=True, batch_dim_first=False):
    """
    Compute the gradients of a neural network 'net' with respect to its input 'x'.

    This function computes the gradients of the network's output 'y' with respect to each input element in 'x'.

    Args:
        net (torch.nn.Module): The neural network for which the gradients are computed.
        x (torch.Tensor): The input data with shape (batch_size x num_inputs).
        create_graph (bool, optional): Whether to create a graph for higher-order derivatives.
        retain_graph (bool, optional): Whether to retain the computation graph.
        allow_unused (bool, optional): Whether to allow unused gradients.
        squeeze_on_one_output (bool, optional): If True and the output dimension of the gradients is 1,
            squeeze the resulting gradients to (batch_size x num_inputs) if 'batch_dim_first' is False,
            or to (num_inputs x batch_size) if 'batch_dim_first' is True.
        batch_dim_first (bool, optional): If True, reshape the gradients as (batch_size, num_inputs, num_outputs).
            If False (default), reshape as (num_outputs, batch_size, num_inputs).

    Returns:
        torch.Tensor: The gradients of the network's output 'y' with respect to the input 'x'.
        The shape depends on the arguments, and it may be adjusted based on 'squeeze_on_one_output' and 'batch_dim_first'.
    """
    x.requires_grad_()
    y = net(x)
    num_out = y.size(1)
    jac = []
    for dim in range(num_out):
        w = torch.zeros_like(y)
        w[:, dim] += 1.0
        grad_x, = grad(y, x, w, retain_graph=retain_graph, create_graph=create_graph, allow_unused=allow_unused)
        jac.append(grad_x)
    jac = torch.stack(jac)
    if jac.size(0) == 1 and squeeze_on_one_output:
        # Squeeze the gradients if the output dimension is 1
        return jac.squeeze(dim=0)

    if batch_dim_first:
        # Reshape the gradients as (batch_size, num_inputs, num_outputs)
        return torch.moveaxis(jac, 0, -1)
    return jac



# Pytorch currently has an implementation of following functions

# def get_vjp_from_net(net, x, v, numpy_output=False):
#     # computes jacobian (dot) v for (numpy or tensor inputs x and v)
#     x = x if x.is_tensor() else torch.tensor(x)
#     jac = get_jacobian_from_net(net, x)
#     if isinstance(v, np.ndarray) and not numpy_output:
#         v = torch.tensor(v, dtype=torch.float32)
#     if numpy_output:
#         v = v if isinstance(v, np.ndarray) else v.numpy()
#         return np.dot(jac.detach().numpy(), v)
#     return torch.dot(jac(net, x), v)


# def Rop(y, x, v):
#     """Computes an Rop.
#
#     Arguments:
#       y (Variable): output of differentiated function
#       x (Variable): differentiated input
#       v (Variable): vector to be multiplied with Jacobian from the right
#     """
#     w = torch.ones_like(y, requires_grad=True)
#     return torch.autograd.grad(torch.autograd.grad(y, x, w), w, v)

# def get_grad(y, x):
#     w = torch.ones_like(y, requires_grad=True)
#     return torch.autograd.grad(y, x, w)


