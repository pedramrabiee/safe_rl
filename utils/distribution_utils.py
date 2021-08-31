import torch
import torch.nn.functional as F

def gaussian_nll_loss(targets, mu, var):
    """ Negative log likelihood function"""
    return 0.5 * (torch.log(var) + torch.div(torch.square(targets - mu), var)).sum()

def bound_log_var(log_var, max_logvar, min_logvar):
    log_var = max_logvar - F.softplus(max_logvar - log_var)
    log_var = min_logvar + F.softplus(log_var - min_logvar)
    return log_var
