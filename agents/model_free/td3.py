import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from agents.model_free.ddpg import DDPGAgent
from networks.mlp import MLPNetwork
from networks.multi_input_mlp import MultiInputMLP
from utils.misc import *
from logger import logger


class TD3Agent(DDPGAgent):
    def initialize(self, params, init_dict=None):
        # get the observation dim from observation process class
        self._obs_dim = self.obs_proc.obs_dim(proc_key='mf')
        if init_dict is not None:
            if 'obs_dim' in init_dict:
                self._obs_dim = init_dict['obs_dim']

        # instantiate policy and critic
        self.policy = MLPNetwork(in_dim=self._obs_dim, out_dim=self._ac_dim,
                                 **params.pi_net_kwargs)

        if params.multi_in_critic:
            self.critics = [MultiInputMLP(in1_dim=self._obs_dim, in2_dim=self._ac_dim, out_dim=1,
                                          **params.q_net_kwargs, **params.multi_in_critic_kwargs) for _ in range(2)]
        else:
            self.critics = [MLPNetwork(in_dim=self._obs_dim + self._ac_dim, out_dim=1,
                                       **params.q_net_kwargs) for _ in range(2)]


        self.policy_target = hard_copy(self.policy)
        self.critic_targets = [hard_copy(critic) for critic in self.critics]

        # freeze target networks
        freeze_net(self.policy_target)
        for critic_target in self.critic_targets:
            freeze_net(critic_target)

        # instantiate optimizers
        self.policy_optimizer = params.pi_optim_cls(self.policy.parameters(), **params.pi_optim_kwargs)
        self.critic_optimizers = [params.q_optim_cls(critic.parameters(), **params.q_optim_kwargs)
                                  for critic in self.critics]


        # instantiate action exploration
        if not self._discrete_action:
            self.exploration = params.exp_strategy_cls(self._ac_dim, **params.exp_kwargs)

        self.models = [self.policy, *self.critics, self.policy_target, *self.critic_targets]
        # TODO: Check model loading and saving
        self.models_dict = dict(policy=self.policy,
                                critic_1=self.critics[0],
                                critic_2=self.critics[1],
                                policy_target=self.policy_target,
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


        # initialize last episode noise is reset and rescaled as -1
        self._reset_noise_ep = -1

        self._optim_timer = 0


    def optimize_agent(self, samples, optim_dict=None):
        # run one gradient descent step for Q
        for critic_optim in self.critic_optimizers:
            critic_optim.zero_grad()

        loss_critic, loss_critics = self._compute_critics_loss(samples)

        loss_critic.backward()

        # TODO: Check
        # clip grad
        if self.params.use_clip_grad_norm:
            for critic in self.critics:
                clip_grad_norm_(critic.parameters(), self.params.clip_max_norm)

        # TODO (optional): critics params can be stacked together to share the same optimizer
        for critic_optim in self.critic_optimizers:
            critic_optim.step()

        policy_trained = False
        if self._optim_timer % self.params.policy_delay == 0:
            # freeze q-network to save computational effort
            for critic in self.critics:
                freeze_net(critic)

            # run one gradient descent for policy
            self.policy_optimizer.zero_grad()
            loss_policy = self._compute_policy_loss(samples)
            loss_policy.backward()
            # clip grad
            if self.params.use_clip_grad_norm:
                clip_grad_norm_(self.policy.parameters(), self.params.clip_max_norm)

            self.policy_optimizer.step()

            # unfreeze q-network
            for critic in self.critics:
                unfreeze_net(critic)

            policy_trained = True

        # count the number of time optimize agent is called
        self._optim_timer += 1

        # Update target networks
        with torch.no_grad():
            polyak_update(target=self.policy_target,
                          source=self.policy,
                          tau=self.params.tau)
            for critic, critic_target in zip(self.critics, self.critic_targets):
                polyak_update(target=critic_target,
                              source=critic,
                              tau=self.params.tau)

        if policy_trained:
            optim_info = {
                "Loss/Policy": loss_policy.cpu().data.numpy(),
                "Loss/Critic1": loss_critics[0].cpu().data.numpy(),
                "Loss/Critic2": loss_critics[1].cpu().data.numpy(),
                "Loss/Critic": loss_critic.cpu().data.numpy()}
        else:
            optim_info = {
                "Loss/Critic1": loss_critics[0].cpu().data.numpy(),
                "Loss/Critic2": loss_critics[1].cpu().data.numpy(),
                "Loss/Critic": loss_critic.cpu().data.numpy()}

        # add log to logger
        logger.add_tabular(optim_info, cat_key='iteration')

        return optim_info

    def _compute_critics_loss(self, sample, critic_loss_dict=None):
        q_vals = [critic(torch.cat((sample.obs, sample.ac), dim=-1)) for critic in self.critics]

        # Bellman backup for Q fucntion
        with torch.no_grad():
            if self._discrete_action:
                raise NotImplementedError
                # target_ac = one_hot_from_logits(self.policy_target(sample.next_obs))
            else:
                target_ac = self.policy_target(sample.next_obs)
                epsilon = torch.rand_like(target_ac) * self.params.target_noise
                epsilon = torch.clamp(epsilon, -self.params.noise_clip_at, self.params.noise_clip_at)
                target_ac = (target_ac + epsilon).clamp(self._ac_lim['low'], self._ac_lim['high'])

            q_pi_targets = [critic_target(torch.cat((sample.next_obs, target_ac), dim=-1))
                            for critic_target in self.critic_targets]
            q_pi_target = torch.hstack(q_pi_targets).min(dim=-1).values.reshape(-1, 1)

            backup = sample.rew + self.params.gamma * (1 - sample.done) * q_pi_target

        # MSE loss against Bellman backup
        loss_critics = [F.mse_loss(q, backup) for q in q_vals]
        loss_critic = loss_critics[0] + loss_critics[1]
        return loss_critic, loss_critics


    def _compute_policy_loss(self, sample, policy_loss_dict=None):
        if self._discrete_action:
            raise NotImplementedError
            # q_pi = self.critics[0](torch.cat((sample.obs, F.gumbel_softmax(self.policy(sample.obs), hard=True)), dim=-1))
        else:
            q_pi = self.critics[0](torch.cat((sample.obs, self.policy(sample.obs)), dim=-1))
        return -q_pi.mean()

