from utils.console import colorize
import numpy as np

seed_ = 2 ** 32 + 475896325
rng = None

def set_seed():
    """Sets random.seed, generate default_rng, torch.manual_seed,
    torch.cuda.manual_seed."""
    global seed_, rng
    seed = seed_ % 4294967290
    seed_ = seed
    import random
    random.seed(seed)
    # for the use of any external packages set numpy random seed
    np.random.seed(seed)    # bad practice to rely on this for creating random numbers (create standalone rng)
    # create random generator
    rng = np.random.default_rng(seed)
    import torch
    torch.backends.cudnn.benchmark = False      # comment this line for performance, uncomment it for reproducibility (not sure if it's essential though)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
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
