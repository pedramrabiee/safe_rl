import torch

from agents.base_agent import BaseAgent
from networks.multi_input_mlp import MultiInputMLP
from networks.mlp import MLPNetwork
from networks.gaussian_mlp import SquashedGaussianMLP
from logger import logger
from utils.misc import *
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class SACAgent(BaseAgent):
    def initialize(self, params, init_dict=None):
        # get the observation dim from observation process class
        self._obs_dim = self.obs_proc.obs_dim(proc_key='mf')
        if init_dict is not None:
            if 'obs_dim' in init_dict:
                self._obs_dim = init_dict['obs_dim']

        self.policy = SquashedGaussianMLP(in_dim=self._obs_dim, out_dim=self._ac_dim,
                                          **params.pi_net_kwargs)

        if params.multi_in_critic:
            self.critics = [MultiInputMLP(in1_dim=self._obs_dim, in2_dim=self._ac_dim, out_dim=1,
                                          **params.q_net_kwargs, **params.multi_in_critic_kwargs) for _ in range(2)]
        else:
            self.critics = [MLPNetwork(in_dim=self._obs_dim + self._ac_dim, out_dim=1,
                                       **params.q_net_kwargs) for _ in range(2)]


        self.critic_targets = [hard_copy(critic) for critic in self.critics]

        # freeze target networks
        for critic_target in self.critic_targets:
            freeze_net(critic_target)

        # instantiate optimizers
        self.policy_optimizer = params.pi_optim_cls(self.policy.parameters(), **params.pi_optim_kwargs)
        self.critic_optimizers = [params.q_optim_cls(critic.parameters(), **params.q_optim_kwargs)
                                  for critic in self.critics]


        self.models = [self.policy, *self.critics, *self.critic_targets]
        # TODO: Check model loading and saving
        self.models_dict = dict(policy=self.policy,
                                critic_1=self.critics[0],
                                critic_2=self.critics[1],
                                critic_target_1=self.critic_targets[0],
                                critic_target_2=self.critic_targets[1]
                                )

        self.optimizers = [self.policy_optimizer, *self.critic_optimizers]
        # TODO: Check optimizer loading and saving
        self.optimizers_dict = dict(policy_optimizer=self.policy_optimizer,
                                    critic_optimizer_1=self.critic_optimizers[0],
                                    critic_optimizer_2=self.critic_optimizers[1])

        self.params = params
        self.init_dict = init_dict

    def step(self, obs, explore=False, init_phase=False):
        # process observation to match the models' input requirement
        obs = self.obs_proc.proc(obs, proc_key='mf')
        obs = torch.as_tensor(obs, dtype=torch.float64)

        # Get action. Action is squashed gaussian, the output is tanh saturated
        action = self.policy(obs, deterministic=not explore, with_log_prob=False)
        # Scale action to ac lim
        action = self._scale_action(action)
        return action, None


    def optimize_agent(self, samples, optim_dict=None):
        # run one gradient descent step for Q
        for critic_optim in self.critic_optimizers:
            critic_optim.zero_grad()

        loss_critic, loss_critics = self._compute_critics_loss(samples)

        loss_critic.backward()

        for critic_optim in self.critic_optimizers:
            critic_optim.step()

        # freeze q-network to save computational effort
        for critic in self.critics:
            freeze_net(critic)

        # run one gradient descent for policy
        self.policy_optimizer.zero_grad()
        loss_policy = self._compute_policy_loss(samples)
        loss_policy.backward()
        self.policy_optimizer.step()

        # unfreeze q-network
        for critic in self.critics:
            unfreeze_net(critic)

        # Update target networks
        with torch.no_grad():
            for critic, critic_target in zip(self.critics, self.critic_targets):
                polyak_update(target=critic_target,
                              source=critic,
                              tau=self.params.tau)

        optim_info = {
            "Loss/Policy": loss_policy.cpu().data.numpy(),
            "Loss/Critic1": loss_critics[0].cpu().data.numpy(),
            "Loss/Critic2": loss_critics[1].cpu().data.numpy(),
            "Loss/Critic": loss_critic.cpu().data.numpy()}

        # add log to logger
        logger.add_tabular(optim_info, cat_key='iteration')

        return optim_info


    def _compute_critics_loss(self, sample, critic_loss_dict=None):
        q_vals = [critic(torch.cat((sample.obs, sample.ac), dim=-1)) for critic in self.critics]

        with torch.no_grad():
            if self._discrete_action:
                raise NotImplementedError

            target_ac, target_ac_log_prob = self.policy(sample.next_obs)
            target_ac = self._scale_action(target_ac)

            q_pi_targets = [critic_target(torch.cat((sample.next_obs, target_ac), dim=-1))
                            for critic_target in self.critic_targets]
            q_pi_target = torch.hstack(q_pi_targets).min(dim=-1).values.reshape(-1, 1)

            backup = sample.rew + self.params.gamma * (1 - sample.done) *\
                     (q_pi_target - self.params.alpha * target_ac_log_prob)

        loss_critics = [F.mse_loss(q, backup) for q in q_vals]
        loss_critic = loss_critics[0] + loss_critics[1]
        return loss_critic, loss_critics

    def _compute_policy_loss(self, sample, policy_loss_dict=None):
        if self._discrete_action:
            raise NotImplementedError

        ac, ac_log_prob = self.policy(sample.obs)
        ac = self._scale_action(ac)

        q_pis = [critic(torch.cat((sample.obs, ac), dim=-1)) for critic in self.critics]
        q_pi = torch.hstack(q_pis).min(dim=-1).values.reshape(-1, 1)

        return (self.params.alpha * ac_log_prob - q_pi).mean()

    def _scale_action(self, action):
        # TODO: check for multi input case
        action = action * self._ac_lim['high']
        # actions should be within the bound, just in case
        return action.clamp(self._ac_lim['low'], self._ac_lim['high'])





