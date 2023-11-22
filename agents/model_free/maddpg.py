import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from agents.model_free.ddpg import DDPGAgent
from distributions.dist_functional import one_hot_from_logits
from networks.mlp import MLPNetwork
from attrdict import AttrDict
from utils.misc import *


class MADDPGAgent(DDPGAgent):
    # This class is copied from pymarl https://github.com/pedramrabiee/pymarl
    # It has been not tested yet
    def initialize(self, params, init_dict=None):

        # instantiate policy and critic
        # FIXME: network inputs
        self.policy = MLPNetwork(in_dim=self._obs_dim, out_dim=self._ac_dim, **params.pi_net_kwargs)
        self.critic = MLPNetwork(in_dim=init_dict['critic_obs_dim'] + init_dict['critic_ac_dim'],
                                 out_dim=1, **params.q_net_kwargs)

        # make target nets
        self.policy_target = hard_copy(self.policy)
        self.critic_target = hard_copy(self.critic)

        # instantiate optimizers
        self.policy_optimizer = params.pi_optim_cls(self.policy.parameters(),  **params.pi_optim_kwargs)
        self.critic_optimizer = params.q_optim_cls(self.critic.parameters(), **params.q_optim_kwargs)

        # instantiate action exploration
        if not self._discrete_action:
            self.exploration = params.exp_strategy_cls(self._ac_dim)

        # list models
        self.models = [self.policy, self.critic, self.policy_target, self.critic_target]
        self.optimizers = [self.policy_optimizer, self.critic_optimizer]

        self.params = params

    def optimize_agent(self, samples, optim_dict=None):
        policies = optim_dict['policies']
        policies_target = optim_dict['policies_target']

        # concatenate agents data  # TODO: change comment
        samples = self._concat_agent_data(samples)
        # run one gradient descent step for Q
        self.critic_optimizer.zero_grad()
        loss_critic = self._compute_critic_loss(samples=samples, policies_target=policies_target)
        loss_critic.backward()
        # clip grad
        if self.params.use_clip_grad_norm:
            clip_grad_norm_(self.critic.parameters(), self.params.clip_max_norm)
        self.critic_optimizer.step()

        # freeze q-network to save computational effort
        freeze_net(self.critic)

        # run one gradient descent for policy
        self.policy_optimizer.zero_grad()
        loss_policy = self._compute_policy_loss(samples=samples, policies=policies)
        loss_policy.backward()

        # clip grad
        if self.params.use_clip_grad_norm:
            clip_grad_norm_(self.policy.parameters(), self.params.clip_max_norm)

        self.policy_optimizer.step()

        # unfreeze q-network
        unfreeze_net(self.critic)

        return {"Loss/Policy": loss_policy.cpu().data.numpy(),
                "Loss/Critic": loss_critic.cpu().data.numpy()}

        # return dict(PolicyLoss=loss_policy.cpu().data.numpy(),
        #             CriticLoss=loss_critic.cpu().data.numpy())

    def _compute_critic_loss(self, samples, policies_target):
        q = self.critic(torch.cat((*samples.obs, *samples.ac), dim=-1))
        # compute target actions for all agents based on target policies and next observations
        with torch.no_grad():
            if self._discrete_action:
                target_acs = [one_hot_from_logits(pi(obs)) for pi, obs in zip(policies_target, samples.next_obs)]
            else:
                target_acs = [pi(obs) for pi, obs in zip(policies_target, samples.next_obs)]
            q_target = self.critic_target(torch.cat((*samples.next_obs, *target_acs), dim=-1))
            backup = samples.rew[self.id] + self.params.gamma * (1 - samples.done[self.id]) * q_target

        # MSE loss against Bellman backup
        loss_critic = F.mse_loss(q, backup)
        return loss_critic

    def _compute_policy_loss(self, samples, policies):
        # acs = [pi(obs) for pi, obs in zip(policies, samples.obs)]
        acs = samples.ac.copy()
        if self._discrete_action:
            acs[self.id] = F.gumbel_softmax(self.policy(samples.obs[self.id]))
        else:
            acs[self.id] = self.policy(samples.obs[self.id])


        q = self.critic(torch.cat((*samples.obs, *acs), dim=-1))
        loss_policy = -q.mean()
        loss_policy += (acs[self.id] ** 2).mean() * 1e-3
        # TODO see MADDPG-Pytorch: he added a reguralizer term to the loss
        return loss_policy

    # helper functions
    @staticmethod
    def _concat_agent_data(samples):
        keys = samples[0].keys()
        num_agent = len(samples)
        samples_dict = AttrDict()
        for key in keys:
            samples_dict[key] = [samples[i][key] for i in range(num_agent)]
        return samples_dict
