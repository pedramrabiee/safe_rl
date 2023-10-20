import numpy as np
from dynamics.nominal_dynamics import NominalDynamics
from envs_utils.misc_env.cbf_test.cbf_test_configs import env_config

class CbfTestDynamics(NominalDynamics):
    def _predict(self, obs, ac, split_return=False):
        c = env_config.c
        k = env_config.k
        m = env_config.m

        x1 = obs[..., 0]
        x2 = obs[..., 1]
        f_func = lambda x1, x2: np.array([x2, - k/m * x1 - c/m * x2], dtype=np.float32)
        G_func = lambda x1, x2: np.array([0.0, 1/m])

        f = np.stack(list(map(f_func, x1, x2)), axis=0)
        G = np.stack(list(map(G_func, x1, x2)), axis=0)
        G = np.expand_dims(G, axis=-1)

        return (f, G) if split_return else f + np.matmul(G, ac).squeeze(-1)