from trainers.base_trainer import BaseTrainer
from attrdict import AttrDict
from logger import logger


class RLBUSTrainer(BaseTrainer):
    def initialize(self):
        self.agent.shield.safe_set.late_initialize(init_dict=AttrDict(backup_agent=self.agent.shield))
        self.sampler.safe_set.late_initialize(init_dict=AttrDict(backup_agent=self.agent.shield))
        self.sampler.safe_set_eval.late_initialize(init_dict=AttrDict(backup_agent=self.agent.shield))

    def _train(self, itr):
        # Pretrain rl backup by sampling desired safe states
        if itr == 0 and self.config.rlbus_params.rl_backup_pretrain_is_on and self.config.rlbus_params.to_shield:
            batch_size = self.config.rlbus_params.rl_backup_pretrain_sample_size
            logger.log('Safe set sampling started...')
            samples = self.agent.shield.safe_set.sample_by_criteria(criteria_keys=['des_safe'],
                                                                    batch_size=[batch_size])

            self.agent.shield.pre_train(self.obs_proc.proc(samples[0], proc_key='rl_backup'))

        # Collect samples
        self.sampler.collect_data(itr)

        # Train
        optim_dict = self._prep_optimizer_dict()
        optim_dict['itr'] = itr

        to_train = []
        samples = {}

        if not self.config.rlbus_params.to_shield:
            self.sampler.collect_data(itr)

        # train desired policy on its train frequency
        self.agent.train_mode(device=self.config.training_device)

        if self.config.rlbus_params.use_mf_desired_policy and \
                itr % self.config.rlbus_params.desired_policy_update_freq == 0 and\
                self.config.rlbus_params.desired_policy_train_in_on:
            to_train.append('desired_policy')
            samples['desired_policy'] =\
                self.sampler.sample(
                    device=self.config.training_device,
                    ow_batch_size=self.config.rlbus_params.desired_policy_train_batch_size)

        # train rl backup
        if itr % self.config.rlbus_params.rl_backup_update_freq == 0 and self.config.rlbus_params.rl_backup_train_is_on \
                and self.config.rlbus_params.to_shield:
            to_train.append('rl_backup')
            batch_size = self.agent.shield.buffer_size() if \
                (self.config.rlbus_params.rl_backup_train_batch_size == 'all') else\
                self.config.rlbus_params.rl_backup_train_batch_size

            random_indices = self.agent.shield.get_random_indices(batch_size)
            samples['rl_backup'] = self.agent.shield.get_samples(random_indices, device=self.config.training_device)

        optim_dict['to_train'] = to_train

        _ = self.agent.optimize_agent(samples, optim_dict)

        self.agent.after_optimize()


