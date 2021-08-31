from trainers.base_trainer import BaseTrainer
from logger import logger
import torch
from utils.misc import torchify
import numpy as np
from utils import scale
from utils.torch_utils import apply_mask_to_dict_of_tensors
from utils.schedule_utils import multi_stage_schedule
from utils.misc import e_or
from attrdict import AttrDict
from time import time
from utils.safe_set import SafeSetFromCriteria, SafeSetFromData


class SFTrainer(BaseTrainer):
    def _train(self, itr):
        # max_speed = self.env.max_speed      # TODO: this is not applicable to all environments. Change this
        version = self.agent.safety_filter.params.deriv_loss_version
        # pretrain safety filter
        if itr == 0 and self.config.sf_params.safety_filter_is_on and self.config.sf_params.filter_pretrain_is_on:
            if issubclass(type(self.safe_set), SafeSetFromCriteria):
                batch_size = self.config.sf_params.filter_initial_training_batch_size

                sample_initial_time = time()
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

                # deriv samples for the deriv loss in pretraining
                deriv_samples = out_cond_unsafe_samples if version == 1 else mid_cond_safe_samples

                print(f'Sampling time: {time() - sample_initial_time}')

                # query dynamics values for the deriv_samples to be used in deriv loss in pretraining
                safe_ac = self.safe_set.get_safe_action(obs=deriv_samples)
                # TODO: this is only implemented with nominal dynamics, add predicted dyanamics to this
                nom_dyn = self.agent.mb_agent.dynamics.predict(obs=deriv_samples,
                                                               ac=safe_ac,
                                                               only_nominal=True,
                                                               stats=None)

                # pretrain filter
                self.agent.pre_train_filter(samples=dict(safe_samples=safe_samples,
                                                         unsafe_samples=unsafe_samples,
                                                         deriv_samples=deriv_samples,
                                                         ),
                                            pre_train_dict=dict(nom_dyn=nom_dyn))

                # logger.dump_plot_with_key(plt_key="loss_plots",
                #                           filename='losses_itr_%d' % itr,
                #                           subplot=False,
                #                           plt_info=dict(
                #                               xlabel=r'Epoch',
                #                               ylabel=r'Loss',
                #                               legend=[r'Safe Loss',
                #                                       r'Unsafe Loss',
                #                                       r'Derivative Loss'],
                #                               yscale='log',
                #                           ),
                #                           plt_kwargs=dict(alpha=0.7),
                #                           columns=['safe loss', 'unsafe loss', 'derivative loss']
                #                           )
                # self.agent.safety_filter.plotter(itr, max_speed)
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
            filter_samples = self.sampler.sample(device=self.config.training_device,
                                                 ow_batch_size=self.config.sf_params.filter_train_batch_size)

            filter_samples = self._obs_proc_from_samples_by_key(filter_samples, proc_key='filter')
            # safe samples
            is_safe_mask = e_or(*self.safe_set.filter_sample_by_criteria(filter_samples.obs, ['in_safe', 'mid_safe', 'out_cond_safe']))
            safe_samples = apply_mask_to_dict_of_tensors(filter_samples, is_safe_mask)

            # unsafe samples
            is_unsafe_mask = e_or(*self.safe_set.filter_sample_by_criteria(filter_samples.obs, ['unsafe', 'out_cond_unsafe']))
            unsafe_samples = apply_mask_to_dict_of_tensors(filter_samples, is_unsafe_mask)

            # deriv samples mask
            if version == 1:
                deriv_mask = self.safe_set.filter_sample_by_criteria(filter_samples.obs, ['out_cond_unsafe'])

            if version == 2:
                deriv_mask = self.safe_set.filter_sample_by_criteria(filter_samples.obs, ['mid_cond_safe'])

            # deriv samples
            deriv_samples = apply_mask_to_dict_of_tensors(filter_samples, deriv_mask)
            if deriv_samples.obs.size(0) > 0:
                safe_ac = self.safe_set.get_safe_action(obs=deriv_samples.obs)
                # TODO: this is only implemented with nominal dynamics, add predicted dyanamics to this
                nom_dyn = self.agent.mb_agent.dynamics.predict(obs=deriv_samples.obs,
                                                               ac=safe_ac,
                                                               only_nominal=True,
                                                               stats=None)
            else:
                nom_dyn = None

            # make safe mask for the samples
            samples['filter'] = dict(safe_samples=safe_samples,
                                     unsafe_samples=unsafe_samples,
                                     deriv_samples=deriv_samples,
                                     dyns=nom_dyn)
            samples['filter'] = AttrDict(**filter_samples, **samples['filter'])

        optim_dict['to_train'] = to_train

        _ = self.agent.optimize_agent(samples, optim_dict)
        # plot
        # if 'filter' in to_train:
            # self.agent.safety_filter.plotter(itr, max_speed)

        logger.dump_tabular(cat_key='iteration', log=False, wandb_log=True, csv_log=False)

        # run after optimize
        self.agent.after_optimize()

