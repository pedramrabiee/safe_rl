from trainers.base_trainer import BaseTrainer
from attrdict import AttrDict
from logger import logger
from functools import partial
import numpy as np



class RLBUSTrainer(BaseTrainer):
    def initialize(self):
        self.agent.shield.safe_set.late_initialize(init_dict=AttrDict(backup_agent=self.agent.shield))
        self.sampler.safe_set.late_initialize(init_dict=AttrDict(backup_agent=self.agent.shield))
        self.sampler.safe_set_eval.late_initialize(init_dict=AttrDict(backup_agent=self.agent.shield))
        self._episode_train_by_sampling = -1


    def _train(self, itr):
        buffer = self.agent.shield.get_buffer()
        self.custom_plotter.dump(itr=itr,
                                 dump_dict=dict(
                                     backup_set_funcs=self.agent.shield.get_backup_sets_for_contour(),
                                     safe_set_func=self.agent.shield.get_safe_set_for_contour(),
                                     viability_kernel_funcs=[partial(self.agent.shield.get_h_per_id_from_batch_of_obs,
                                                                     id=idx)
                                                             for idx in range(self.agent.shield.backup_set_size)],
                                     buffer_data=buffer.obs
                                 ))

        if self.config.rlbus_params.train_by_sampling_from_state_space and\
                self.sampler.episode_completed > self._episode_train_by_sampling and\
                self.sampler.episode_completed > -1:
            batch_size = self.config.rlbus_params.train_by_sampling_from_state_space_batch_size

            # samples = self.agent.shield.safe_set.sample_by_criteria(criteria_keys=['safe'],
            #                                                         batch_size=[batch_size])

            samples = self.agent.shield.safe_set.sample_by_criteria(criteria_keys=['near_boundary', 'safe'],
                                                                    batch_size=[batch_size, int(batch_size/4)])

            self.agent.shield.add_batch_of_data_to_buffer_from_obs(obs=self.obs_proc.proc(np.vstack((samples[0], samples[1])),
                                                                                          proc_key='shield'))

            self._episode_train_by_sampling += 1


        # PRETRAIN rl backup by sampling desired safe states
        if itr == 0 and self.config.rlbus_params.rl_backup_pretrain_is_on and self.config.rlbus_params.to_shield:
            self.agent.shield.include_rl_backup_in_h = False
            batch_size = self.config.rlbus_params.rl_backup_pretrain_sample_size
            logger.log('Safe set sampling started...')
            samples = self.agent.shield.safe_set.sample_by_criteria(criteria_keys=['safe'],
                                                                    batch_size=[batch_size])

            self.agent.shield.pre_train(self.obs_proc.proc(samples[0], proc_key='rl_backup'))

            self.agent.shield.include_rl_backup_in_h = True

            buffer = self.agent.shield.get_buffer()
            self.custom_plotter.dump(dump_key="h_contours",
                                     dump_dict=dict(
                                         backup_set_funcs=self.agent.shield.get_backup_sets_for_contour(),
                                         safe_set_func=self.agent.shield.get_safe_set_for_contour(),
                                         viability_kernel_funcs=[partial(self.agent.shield.get_h_per_id_from_batch_of_obs,
                                                                         id=idx)
                                                                 for idx in range(self.agent.shield.backup_set_size)],
                                         buffer_data=buffer.obs
                                     ))



        # COLLECT SAMPLES
        # Set rl backup exploration on
        # if self.sampler.episode_completed >= self.config.rlbus_params.rl_backup_explore_episode_delay:

        # FIXME: shield's backup rl is set to explore. However, during the safe reset, it shouldn't use the explore mode.
        self.sampler.collect_data(itr)

        if self.config.rlbus_params.to_shield:
            self.agent.shield.compute_ac_push_to_buffer(self.sampler.episode_completed)

        # TRAIN
        optim_dict = self._prep_optimizer_dict()
        optim_dict['itr'] = itr

        to_train = []
        samples = {}

        # train desired policy on its train frequency
        self.agent.train_mode(device=self.config.training_device)

        for _ in range(self.config.rlbus_params.net_updates_per_iter):
            if self.config.rlbus_params.use_mf_desired_policy and \
                    itr % self.config.rlbus_params.desired_policy_update_freq == 0 and\
                    self.config.rlbus_params.desired_policy_train_in_on:
                to_train.append('desired_policy')
                samples['desired_policy'] =\
                    self.sampler.sample(
                        device=self.config.training_device,
                        ow_batch_size=self.config.rlbus_params.desired_policy_train_batch_size)
                logger.dump_tabular(cat_key='des_policy_iteration', wandb_log=True)


            # train rl backup
            if itr % self.config.rlbus_params.rl_backup_update_freq == 0 and self.config.rlbus_params.rl_backup_train_is_on \
                    and self.config.rlbus_params.to_shield and\
                    self.sampler.episode_completed >= self.config.rlbus_params.episode_to_start_training_rl_backup:

                to_train.append('rl_backup')
                batch_size = self.agent.shield.buffer_size() if \
                    (self.config.rlbus_params.rl_backup_train_batch_size == 'all') else\
                    self.config.rlbus_params.rl_backup_train_batch_size

                random_indices = self.agent.shield.get_random_indices(batch_size)
                samples['rl_backup'] = self.agent.shield.get_samples(random_indices, device=self.config.training_device)
                # samples['rl_backup'] = self.agent.shield.get_customized_samples(batch_size=batch_size, device=self.config.training_device)

                logger.dump_tabular(cat_key='rl_backup_iteration', wandb_log=True)

            optim_dict['to_train'] = to_train

            _ = self.agent.optimize_agent(samples, optim_dict)

            self.agent.after_optimize()


