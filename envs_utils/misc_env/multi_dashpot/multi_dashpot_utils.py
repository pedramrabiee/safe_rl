import numpy as np
from agents.model_based.cbf_test import CBFTESTAgent
from envs_utils.misc_env.multi_dashpot.multi_dashpot_configs import env_config

class MultiDashpotAgent(CBFTESTAgent):
    env_config = env_config
    def _get_lqr_weights(self, A, Ai, B):
        R1 = np.block([
            [10 * np.eye(A.shape[0]), np.zeros((A.shape[0], Ai.shape[0]), dtype=np.float32)],
            [np.zeros((Ai.shape[0], A.shape[0]), dtype=np.float32), 100 * np.eye(Ai.shape[0])]
        ])
        R2 = 100 * np.eye(B.shape[1])
        return R1, R2


