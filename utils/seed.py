from utils.console import colorize
seed_ = None
import numpy as np

def set_seed(seed):
    """Sets random.seed, np.random.seed, torch.manual_seed,
    torch.cuda.manual_seed."""
    seed %= 4294967290
    global seed_
    seed_ = seed
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(colorize(f"using seed {seed}", "green"))
    return seed

def set_env_seed(env, seed):
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)


def get_seed():
    return seed_