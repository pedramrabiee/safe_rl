from dynamics.base_dynamics import BaseDynamics
from networks.mlp import MLPNetwork
import torch.nn as nn
from utils.misc import *
from torch.utils.data import TensorDataset, DataLoader

mse_loss = nn.MSELoss()


class DeterministicDynamics(BaseDynamics):
    def initialize(self, params, init_dict=None):
        super().initialize(params, init_dict)
        if self.use_custom_dyn:
            self.model = self.params.custom_dyn_cls(base_net_cls=MLPNetwork,
                                                    obs_dim=self.obs_dim,
                                                    ac_dim=self.ac_dim,
                                                    net_kwargs=self.params.dynamics_net_kwargs)
        else:
            self.model = MLPNetwork(in_dim=self.obs_dim+self.ac_dim,
                                    out_dim=self.out_dim,
                                    **self.params.dynamics_net_kwargs)

        self.optimizer = self.params.optim_cls(self.model.parameters(), **self.params.optim_kwargs)

    @torch.no_grad()
    def _predict(self, obs, ac, stats, pred_dict=None, split_return=False):
        if split_return:
            dyn_bd = self.model(torch.cat((obs, ac), dim=-1), split_return=True)
            return [x.detach().numpy() for x in dyn_bd]
        if self.params.normalized_io:
            obs_norm = normalize(data=obs, **stats.obs)
            ac_norm = normalize(data=ac, **stats.ac)
            delta_norm = self.model(torch.cat((obs_norm, ac_norm), dim=-1)).detach().numpy()
            delta_unnorm = unnormalize(data=delta_norm, **stats.delta_obs)
            return delta_unnorm
        else:
            delta = self.model(torch.cat((obs, ac), dim=-1)).detach().numpy()
            return delta

    @torch.no_grad()
    def eval(self):
        return self._compute_loss(*self.valid_dataset).cpu().data.numpy()

    def _train(self, itr):
        losses = []
        for inputs, targets in self.train_generator:
            self.optimizer.zero_grad()
            loss = self._compute_loss(inputs, targets)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.cpu().data.numpy())
        return np.stack(losses).mean()

    def _compute_loss(self, inputs, targets):
        return mse_loss(targets, self.model(inputs))

    def _train_preproc(self, samples, stats):
        num_data = samples.rew.shape[0]
        holdout_ratio = self.params.holdout_ratio
        train_ids, valid_ids = train_valid_split(data_size=num_data, holdout_ratio=holdout_ratio)
        train_split, valid_split = self._samples2train_valid(samples=samples,
                                                             train_ids=train_ids,
                                                             valid_ids=valid_ids)

        train_dataset = TensorDataset(*self._samples2io(train_split, stats, is_normalized=self.params.normalized_io))
        self.train_generator = DataLoader(dataset=train_dataset,
                                          batch_size=self.params.batch_size,
                                          shuffle=True)
        self.valid_dataset = self._samples2io(valid_split, stats, is_normalized=self.params.normalized_io)
