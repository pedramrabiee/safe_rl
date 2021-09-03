from trainers.base_trainer import BaseTrainer
from logger import logger
import torch
from utils.misc import torchify
import numpy as np
from utils import scale
from utils.misc import np_object2dict
from utils.schedule_utils import multi_stage_schedule
from utils.misc import e_or
from attrdict import AttrDict
from time import time
from utils.safe_set import SafeSetFromCriteria, SafeSetFromData


class SFTrainer(BaseTrainer):
    def initialize(self):
        # switch buffer to buffer 1
        self.agent.curr_buf_id = 1
        self.agent.buffer.initialize(attribute_names=['safe_samples', 'unsafe_samples',
                                                      'deriv_samples', 'dyn_safe',
                                                      'obs', 'next_obs', 'dyn_values'],
                                     coupled_list=[['deriv_samples', 'dyn_safe'],
                                                   ['obs', 'next_obs', 'dyn_values']]
                                     )
        # switch back buffer to buffer 0
        self.agent.curr_buf_id = 0


    def _train(self, itr):
        version = self.agent.safety_filter.params.deriv_loss_version
        # pretrain safety filter
        if itr == 0 and self.config.sf_params.safety_filter_is_on and self.config.sf_params.filter_pretrain_is_on:
            if issubclass(type(self.safe_set), SafeSetFromCriteria):
                batch_size = self.config.sf_params.filter_initial_training_batch_size
                timer_begin = time()
                # sample safe datasets
                samples = self.safe_set.sample_by_criteria(criteria_keys=['in_safe',
                                                                          'mid_safe',
                                                                          "out_cond_safe",
                                                                          'unsafe',
                                                                          'out_cond_unsafe',
                                                                          "mid_cond_safe"
                                                                          ],
                                                           batch_size=[int(batch_size / 3),
                                                                       int(batch_size / 3),
                                                                       int(batch_size / 3),
                                                                       int(2 * batch_size / 3),
                                                                       int(batch_size / 3),
                                                                       int(batch_size / 3)
                                                                       ])

                safe_in_samples, safe_mid_samples, out_cond_safe_samples,\
                    unsafe_samples, out_cond_unsafe_samples, mid_cond_safe_samples = samples

                # stack safe datasets for the safe loss in pretraining
                safe_samples = np.vstack((safe_in_samples, safe_mid_samples, out_cond_safe_samples))

                # stack unsafe datasets for the unsafe loss in pretraining
                unsafe_samples = np.vstack((unsafe_samples, out_cond_unsafe_samples))

                # deriv experience for the deriv loss in pretraining
                if version == 1:
                    deriv_samples = out_cond_unsafe_samples
                if version == 2:
                    deriv_samples = mid_cond_safe_samples
                if version == 3:
                    deriv_samples = safe_samples

                logger.log(f'Safe set sampling time: {time() - timer_begin}', color='blue')

                # query dynamics values for the deriv_samples to be used in deriv loss in pretraining
                safe_ac = self.safe_set.get_safe_action(obs=deriv_samples) * self.config.cbf_params.u_max_weight_in_deriv_loss
                # TODO: this is only implemented with nominal dynamics, add predicted dyanamics to this
                nom_dyn = self.agent.mb_agent.dynamics.predict(obs=deriv_samples,
                                                               ac=safe_ac,
                                                               only_nominal=True,
                                                               stats=None)


                # add safe and unsafe samples to buffer
                samples = dict(safe_samples=safe_samples,
                               unsafe_samples=unsafe_samples,
                               deriv_samples=deriv_samples,
                               dyn_safe=nom_dyn)

                if self.config.sf_params.add_cbf_pretrain_data_to_buffer:
                    self.agent.curr_buf_id = 1
                    self.agent.buffer.push_to_buffer(experience=samples)
                    self.agent.curr_buf_id = 0

                # pretrain filter
                timer_begin = time()
                self.agent.pre_train_filter(samples=AttrDict(torchify(samples, device=self.config.training_device)))
                logger.log(f'Pretraining time: {time() - timer_begin}', color='blue')

                # dump loss plots
                logger.dump_plot_with_key(plt_key="loss_plots",
                                          filename='losses_itr_%d' % itr,
                                          subplot=False,
                                          plt_info=dict(
                                              xlabel=r'Epoch',
                                              ylabel=r'Loss',
                                              legend=[r'Safe Loss',
                                                      r'Unsafe Loss',
                                                      r'Derivative Loss'],
                                              yscale='log',
                                          ),
                                          plt_kwargs=dict(alpha=0.7),
                                          columns=['safe loss', 'unsafe loss', 'derivative loss']
                                          )
                if hasattr(self.custom_plotter, 'h_plotter'):
                    self.custom_plotter.h_plotter(itr, self.agent.safety_filter.filter_net)
            elif issubclass(type(self.safe_set), SafeSetFromData):
                pass

        # collect data by running current policy
        self.sampler.collect_data(itr)

        # train
        self.agent.train_mode(device=self.config.training_device)
        optim_dict = self._prep_optimizer_dict()
        optim_dict['itr'] = itr

        to_train = []
        samples = {}
        # train model-free on its train frequency
        if itr % self.config.sf_params.mf_update_freq == 0 and self.config.sf_params.mf_train_is_on:
            to_train.append('mf')
            samples['mf'] = self.sampler.sample(device=self.config.training_device,
                                                ow_batch_size=self.config.sf_params.mf_train_batch_size)
            samples['mf'] = self._obs_proc_from_samples_by_key(samples['mf'], proc_key='mf')

        # train model-based on its train frequency
        if itr % self.config.sf_params.mb_update_freq == 0 and self.config.sf_params.dyn_train_is_on:
            to_train.append('mb')
            samples['mb'] = self.sampler.sample(device=self.config.training_device,
                                                ow_batch_size=self.config.sf_params.mb_train_batch_size)

            samples['mb'] = self._obs_proc_from_samples_by_key(samples['mb'], proc_key='mb')

        # train filter on its frequency
        if self.config.sf_params.safety_filter_is_on and self.config.sf_params.filter_train_is_on and\
                multi_stage_schedule(itr=itr, stages_dict=self.config.sf_params.filter_training_stages):
            to_train.append('filter')

            # prior to sampling from buffer, release and process items in the queue and add them to buffer
            # switch to buffer 1
            self.agent.curr_buf_id = 1
            queue_items = self.agent.buffer.release_queue(to_tensor=False)
            queue_items_dyn = np_object2dict(queue_items.info).dyn_out
            queue_items = self._obs_proc_from_samples_by_key(queue_items, proc_key='filter')

            # safe experience
            is_safe_mask = e_or(
                *self.safe_set.filter_sample_by_criteria(queue_items.obs, ['in_safe', 'mid_safe', 'out_cond_safe']))
            safe_samples = queue_items.obs[np.asarray(is_safe_mask), ...]

            # unsafe experience
            is_unsafe_mask = e_or(
                *self.safe_set.filter_sample_by_criteria(queue_items.obs, ['unsafe', 'out_cond_unsafe']))
            unsafe_samples = queue_items.obs[np.asarray(is_unsafe_mask), ...]

            # deriv experience mask
            if version == 1:
                deriv_mask = self.safe_set.filter_sample_by_criteria(queue_items.obs, ['out_cond_unsafe'])

            if version == 2:
                deriv_mask = self.safe_set.filter_sample_by_criteria(queue_items.obs, ['mid_cond_safe'])

            if version == 3:
                deriv_mask = is_safe_mask

            # deriv experience
            deriv_samples = queue_items.obs[np.asarray(deriv_mask), ...]
            if deriv_samples.shape[0] > 0:
                safe_ac = self.safe_set.get_safe_action(obs=deriv_samples) * self.config.cbf_params.u_max_weight_in_deriv_loss
                # TODO: this is only implemented with nominal dynamics, add predicted dyanamics to this
                nom_dyn = self.agent.mb_agent.dynamics.predict(obs=deriv_samples,
                                                               ac=safe_ac,
                                                               only_nominal=True,
                                                               stats=None)
            else:
                nom_dyn = None
            self.agent.buffer.push_to_buffer(experience=dict(safe_samples=safe_samples,
                                                             unsafe_samples=unsafe_samples,
                                                             deriv_samples=deriv_samples,
                                                             dyn_safe=nom_dyn,
                                                             obs=queue_items.obs,
                                                             next_obs=queue_items.next_obs,
                                                             dyn_values=queue_items_dyn))

            filter_samples = self.sampler.sample(device=self.config.training_device,
                                                 ow_batch_size=self.config.sf_params.filter_train_batch_size)

            # make safe mask for the experience
            samples['filter'] = AttrDict(filter_samples)
            # switch back to buffer 0
            self.agent.curr_buf_id = 0


        optim_dict['to_train'] = to_train

        _ = self.agent.optimize_agent(samples, optim_dict)
        # plot h curve
        if 'filter' in to_train and hasattr(self.custom_plotter, 'h_plotter'):
            self.custom_plotter.h_plotter(itr, self.agent.safety_filter.filter_net)

        logger.dump_tabular(cat_key='iteration', log=False, wandb_log=True, csv_log=False)

        # run after optimize
        self.agent.after_optimize()

