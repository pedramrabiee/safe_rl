import numpy as np
from envs_utils.misc_env.cbf_test.cbf_test_configs import env_config
from utils import scale
from agents.base_agent import BaseAgent
from scipy.linalg import block_diag
from control.matlab import *
from logger import logger
from envs_utils.misc_env.cbf_test.cbf_test_env import get_dynamics_matrices

class CBFTESTAgent(BaseAgent):
    env_config = env_config
    def initialize(self, params, init_dict=None):
        self.m = self.env_config.m
        self.k = self.env_config.k
        self.c = self.env_config.c
        self.max_u = self.env_config.max_u
        self.max_speed = self.env_config.max_speed
        self.omega = self.env_config.omega

        # get the observation dim from observation process class
        self._obs_dim = self.obs_proc.obs_dim(proc_key='mf')
        self.params = params

        A, B, C, D = get_dynamics_matrices()
        D1 = B
        le = C.shape[0]

        # LQR DESIGN FOR SINUSOID COMMAND
        I = np.array([[0, -1], [1, 0]])
        Aip = block_diag(0.0, self.omega * np.pi * I)
        self.imp_state_dim = Aip.shape[0]
        Ai = np.kron(np.eye(le), Aip)

        Bip = np.ones((self.imp_state_dim, 1), dtype=np.float32)
        Bi = np.kron(np.eye(le), Bip)

        # Augmented Matrices
        Aa = np.block([
            [A, np.zeros((A.shape[0], Ai.shape[1]), dtype=np.float32)],
            [np.matmul(-Bi, C), Ai]
        ])
        Ba = np.block([
            [B],
            [np.matmul(-Bi, D)]
        ])

        R1, R2 = self._get_lqr_weights(A, Ai, B)

        # LQR Gain
        Ka, S, E = lqr(Aa, Ba, R1, R2)
        Ka = -Ka.A  # .A return array

        self.K = Ka[:, :A.shape[0]]
        self.Ki = Ka[:, A.shape[0]:]

        sys_c = ss(Ai, Bi, np.eye(Ai.shape[0]), np.zeros((self.imp_state_dim, 1)))
        self.sys_cd = c2d(sys_c, self._timestep)
        self.xi = np.zeros(self.imp_state_dim, dtype=np.float32)
        self.t = 0.0
        self.models = []
        self.optimizers = []
        self.models_dict = {}
        self.optimizers_dict = {}
        self.extra_params_dict = {}

    def _get_lqr_weights(self, A, Ai, B):
        R1 = np.block([
            [10 * np.eye(A.shape[0]), np.zeros((A.shape[0], Ai.shape[0]), dtype=np.float32)],
            [np.zeros((Ai.shape[0], A.shape[0]), dtype=np.float32), 100 * np.eye(Ai.shape[0])]
        ])
        R2 = 100 * np.eye(B.shape[1])
        return R1, R2

    def step(self, obs, explore=False, init_phase=False):
        # process observation to match the models' input requirement
        obs = self.obs_proc.proc(obs, proc_key='mf')
        obs = obs.squeeze()
        # action = rng.uniform(-1.0, 1.0, (1, self._ac_dim))
        command = self.env_config.command_amplitude * np.sin(self.omega * np.pi * self.t) + 4
        logger.push_plot(command.reshape(1, -1), plt_key='performance')
        self.t += self._timestep
        self.xi = self.sys_cd.A.A @ self.xi + (self.sys_cd.B.A * (command - obs[0])).squeeze()
        action = self.K @ obs + self.Ki @ self.xi
        # action = np.clip(action, -self.max_u, self.max_u)
        action = scale.action2newbounds(action)
        return action, None

    def on_episode_reset(self, episode):
        self.t = 0.0
        self.xi = np.zeros(self.imp_state_dim, dtype=np.float32)
