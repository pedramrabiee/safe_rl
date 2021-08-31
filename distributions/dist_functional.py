import torch
import numpy as np

def one_hot_from_logits(logits, eps=0.0):
    # max method returns torch.return_type.max object with values and indices attribute
    argmax_acs_mask = (logits == logits.max(dim=-1, keepdim=True).values).float()
    # get one hot encoded best action
    if eps == 0.0:
        return argmax_acs_mask
    # get random actions in one-hot form
    rand_acs = torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]]
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs_mask[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])
