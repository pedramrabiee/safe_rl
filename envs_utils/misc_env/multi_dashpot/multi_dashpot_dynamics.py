import numpy as np
from dynamics.nominal_dynamics import NominalDynamics
from multi_dashpot_env import get_dynamics_matrices


class MultiDashpotDynamics(NominalDynamics):
    def _predict(self, obs, ac, split_return=False):
        A, B, _, _  = get_dynamics_matrices()
        f = (A @ obs.T).T
        G = np.tile(B, (obs.shape[0], 1, 1))
        return (f, G) if split_return else f + np.matmul(G, ac).squeeze(-1)