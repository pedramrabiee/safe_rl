import torch

def nll_loss(y, mu, var):
    # Negative log likelihood function
    return 0.5 * (torch.log(var) + torch.div(torch.square(y - mu), var)).mean()
