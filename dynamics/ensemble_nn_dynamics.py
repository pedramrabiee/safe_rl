from dynamics.base_dynamics import BaseDynamics
import numpy as np
import torch
from utils.misc import train_valid_split
from torch.utils.data import TensorDataset, DataLoader
from utils.seed import rng

class EnsembleDynamics(BaseDynamics):
    def initialize(self, params, init_dict=None):
        super().initialize(params, init_dict)
        self.ensemble_size = self.params.ensemble_size
        # instantiate models in ensemble
        self.ensemble = [init_dict.dynamics_cls(obs_dim=self.obs_dim,
                                                ac_dim=self.ac_dim,
                                                out_dim=self.out_dim,
                                                obs_proc=self.obs_proc) for _ in range(self.ensemble_size)]
        for dyn in self.ensemble:
            dyn.initialize(self.params, init_dict=init_dict)

    @torch.no_grad()
    def _predict(self, obs, ac, stats, pred_dict=None, split_return=False):
        delta = []
        for i, dyn in enumerate(self.ensemble):
            delta.append(dyn._predict(obs[i], ac, stats))

        delta = np.stack(delta, axis=0)

        if pred_dict is None:
            return delta.mean(axis=0) + rng.normal(loc=0, scale=1, size=delta.shape[-1]) * delta.std(axis=0)
        elif pred_dict['return_all']:
            return delta

    def _train(self, itr):
        losses = []
        for dyn in self.ensemble:
            loss = dyn._train(itr)
            losses.append(loss)
        return np.stack(losses).mean()

    def _train_preproc(self, samples, stats):
        num_data = samples.rew.shape[0]
        holdout_ratio = self.params.holdout_ratio
        train_ids, valid_ids = train_valid_split(data_size=num_data, holdout_ratio=holdout_ratio)
        train_splits, valid_split = self._samples2train_valid(samples=samples,
                                                             train_ids=train_ids,
                                                             valid_ids=valid_ids,
                                                             ensemble_size=self.ensemble_size)
        for i, dyn in enumerate(self.ensemble):
            train_dataset = TensorDataset(
                *self._samples2io(train_splits[i], stats, is_normalized=self.params.normalized_io))
            dyn.train_generator = DataLoader(dataset=train_dataset,
                                             batch_size=self.params.batch_size,
                                             shuffle=True)
            dyn.valid_dataset = self._samples2io(valid_split, stats, is_normalized=self.params.normalized_io)

    @torch.no_grad()
    def eval(self):
        losses = []
        for dyn in self.ensemble:
            losses.append(dyn.eval())
        return np.stack(losses).mean()

