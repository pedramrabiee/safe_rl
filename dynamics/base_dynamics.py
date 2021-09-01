from networks.mlp import MLPNetwork
import numpy as np
import torch.nn as nn
import torch
from utils.misc import to_device, normalize, unnormalize
from logger import logger
from attrdict import AttrDict

class BaseDynamics:
    def __init__(self, obs_dim, ac_dim, out_dim, timestep, obs_proc=None):
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.out_dim = out_dim
        self.timestep = timestep
        self.obs_proc = obs_proc

    def initialize(self, params, init_dict=None):
        self.params = params
        self.env_bounds = init_dict.env_bounds

        self.use_custom_dyn = self.params.use_custom_dyn
        if self.params.use_nominal_dyn:
            self.use_nominal_dyn = True
            self.nominal_dyn = self.params.nominal_dyn_dict['cls'](obs_dim=self.obs_dim,
                                                                   ac_dim=self.ac_dim,
                                                                   out_dim=self.out_dim,
                                                                   timestep=self.timestep,
                                                                   env_bounds=self.env_bounds)
            self.nominal_dyn.initialize(self.params.nominal_dyn_dict['params'],
                                        init_dict=dict(is_continuous=self.params.train_continuous_time))
        else:
            self.use_nominal_dyn = False
            self.nominal_dyn = None

    @torch.no_grad()
    def predict(self, obs, ac, stats, pred_dict=None, split_return=False, only_nominal=False):
        # process observation to match the models' input requirement
        obs = self.obs_proc.proc(obs, proc_key='mb')

        # convert obs and ac to torch tensors
        obs_torch = torch.as_tensor(obs, dtype=torch.float32)
        ac_torch = torch.as_tensor(ac, dtype=torch.float32)

        # Case 1: split_return = True
        if split_return:
            # Case 1.1: only_nominal = True
            if only_nominal:
                nom_dyn = self.nominal_dyn.predict(obs, ac, split_return=True)
                return AttrDict(nom_dyn=nom_dyn)
            # Case 1.2: only_nominal = False
            else:
                pred = self._predict(obs_torch, ac_torch, stats, pred_dict, split_return=True)
                # in the discrete-time case the values returned by _predict method when split_return=True is the value of
                # functions used to reproduce delta_obs. You need to add obs to mu_f (in gaussian case) or f
                # (in deterministic case) to have the correct values for reproduction of next_obs
                # in continuous-time case you need to divide pred by t_s. In this case make sure that in probablistic cases,
                # your _predict model returns std instead of var when split_return=True
                if self.params.train_continuous_time:
                    pred = [x / self.timestep for x in pred]     # this is correct only when the probablistic model return std and not var # FIXME: fix this
                else:
                    pred[0] += obs      # this is correct only when f or mu_f is the first element or pred #TODO: Fix this

                nom_dyn = self.nominal_dyn.predict(obs, ac, split_return=True) if self.use_nominal_dyn else None
                return AttrDict(pred_dyn=pred, nom_dyn=nom_dyn)
        # Case 2: split_return = False, only_nominal = True
        if only_nominal:
            return self.nominal_dyn.predict(obs, ac)

        # Case 3: split_return = False, only_nominal = False
        if self.params.train_continuous_time:
            # in this case, the output is obs_dot
            output = self._predict(obs_torch, ac_torch, stats, pred_dict)
        else:
            # in this case, the output is next_obs
            output = self._predict(obs_torch, ac_torch, stats, pred_dict) + obs

        # Add the dynamics value from the nominal dynamics
        output = output + self.nominal_dyn.predict(obs, ac) if self.use_nominal_dyn else output
        return output

    @torch.no_grad()
    def _predict(self, obs, ac, stats, pred_dict=None, split_return=False):
        raise NotImplementedError

    def train(self, samples, train_dict=None):
        stats = train_dict.stats
        itr = train_dict.itr
        # preprocess experience, _preproc will take care of nominal dynamics
        self._train_preproc(samples, stats)

        # run training loop
        for epoch in range(self.params.epochs):
            loss = self._train(itr)
            # log
            if loss is not None:
                logger.add_tabular({"Loss/Dynamics": loss}, cat_key="dynamics_epoch")
                logger.dump_tabular(cat_key="dynamics_epoch", log=False, wandb_log=True, csv_log=False)
        return loss

    def eval_mode(self, device='cpu'):
        """Switch neural net model to evaluation mode"""
        if hasattr(self, 'ensemble'):
            for dyn in self.ensemble:
                dyn.model.eval()
                dyn.model = to_device(dyn.model, device)
        else:
            self.model.eval()
            self.model = to_device(self.model, device)
            if hasattr(self, 'extra_params'):
                for param in self.extra_params:
                    param.requires_grad_(False)
                    param = to_device(param, device)

        if hasattr(self, 'likelihood'):
            self.likelihood.eval()
            self.likelihood = to_device(self.likelihood, device)


    def train_mode(self, device='cpu'):
        """Switch neural net model to training mode"""
        if hasattr(self, 'ensemble'):
            for dyn in self.ensemble:
                dyn.model.train()
                dyn.model = to_device(dyn.model, device)
        else:
            self.model.train()
            self.model = to_device(self.model, device)
            if hasattr(self, 'extra_params'):
                for param in self.extra_params:
                    param.requires_grad_(True)
                    param = to_device(param, device)

        if hasattr(self, 'likelihood'):
            self.likelihood.train()
            self.likelihood = to_device(self.likelihood, device)

    @torch.no_grad()
    def eval(self):
        raise NotImplementedError

    def _train_preproc(self, *args):
        pass

    def _train(self, *args):
        raise NotImplementedError

    def _samples2io(self, samples, stats, is_normalized=False):
        obs = self.obs_proc.proc(samples.obs)
        next_obs = self.obs_proc.proc(samples.next_obs)
        ac = samples.ac
        output = next_obs - obs
        if self.params.train_continuous_time:
            output /= self.timestep

        # prior to normalizing observation and action, get query from the nominal dynamics if self.use_nominal_dyn = True
        if self.use_nominal_dyn:
            output -= self.nominal_dyn.predict(obs, ac)

        if is_normalized:
            obs = normalize(data=obs, **stats.obs)          # stats data is already preprocessed in the get_stats method of buffer
            ac = normalize(data=ac, **stats.ac)
            delta_obs_stats = stats.delta_obs
            if self.params.train_continuous_time:
                delta_obs_stats.mean -= self.nominal_dyn.predict(obs, ac) * self.timestep
            output = normalize(data=output, **delta_obs_stats)      # TODO: this needs to be corrected for nominal_dyn: DONE, just check

        inputs = torch.cat((obs, ac), dim=-1)

        return inputs, output

    @staticmethod
    def _samples2train_valid(samples, train_ids, valid_ids, ensemble_size=None):
        train_data = {}
        valid_data = {}

        if ensemble_size is None:
            for k in samples.keys():
                train_data[k] = samples[k][train_ids, :] if k != 'info' else samples[k][train_ids]
                valid_data[k] = samples[k][valid_ids, :] if k != 'info' else samples[k][valid_ids]
            return AttrDict(train_data), AttrDict(valid_data)

        else:
            for k in samples.keys():
                valid_data[k] = samples[k][valid_ids, :] if k != 'info' else samples[k][valid_ids]

            train_splits = []
            num_train = len(train_ids)
            for i in range(ensemble_size):
                idx = np.random.choice(num_train, num_train, replace=True)
                for k in samples.keys():
                    train_data[k] = samples[k][idx, :] if k != 'info' else samples[k][idx]
                train_splits.append(AttrDict(train_data))

            return train_splits, AttrDict(valid_data)

