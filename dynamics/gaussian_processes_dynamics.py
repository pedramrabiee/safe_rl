import torch
import gpytorch
from dynamics.base_dynamics import BaseDynamics
from networks.gaussian_processes import GP
from utils.misc import train_valid_split, normalize, unnormalize
from dynamics.affine_in_action import AffineInActionGP

class GPDynamics(BaseDynamics):
    def initialize(self, params, init_dict=None):
        super().initialize(params, init_dict)

        if self.use_custom_dyn:
            if self.params.custom_dyn_cls == AffineInActionGP:
                self.out_dim = self.obs_dim * (1 + self.ac_dim)
                self._instantiate_likelihood()
            self.model = self.params.custom_dyn_cls(obs_dim=self.obs_dim,
                                                    ac_dim=self.ac_dim,
                                                    net_kwargs=dict(likelihood=self.likelihood))
        else:
            self._instantiate_likelihood()
            self.model = GP(in_dim=self.obs_dim + self.ac_dim,
                            out_dim=self.out_dim,
                            likelihood=self.likelihood)

        self.optimizer = self.params.optim_cls(self.model.parameters(), **self.params.optim_kwargs)
        self.is_trained = False
        self.mll_loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    @torch.no_grad()
    def _predict(self, obs, ac, stats, pred_dict=None, split_return=False):
        assert self.is_trained, "GP needs to be trained before making any prediction."
        if self.params.normalized_io:
            obs_norm = normalize(data=obs, **stats.obs)
            ac_norm = normalize(data=ac, **stats.ac)
            with gpytorch.settings.fast_pred_var():
                # delta_norm = self.likelihood(self.model(torch.cat((obs_norm, ac_norm), dim=-1))).mean.detach().numpy() # uncomment to use mean values
                delta_norm = self.likelihood(self.model(torch.cat((obs_norm, ac_norm), dim=-1))).sample().detach().numpy()
                delta_unnorm = unnormalize(data=delta_norm, **stats.delta_obs) # TODO: check if it is correct to do this with experience, it used to be with mean
            return delta_unnorm
        else:
            with gpytorch.settings.fast_pred_var():
                # delta = self.likelihood(self.model(torch.cat((obs_torch, ac_torch), dim=-1))).mean.detach().numpy() # uncomment to use mean values
                delta = self.model(torch.cat((obs, ac), dim=-1), ret_sample=True).detach().numpy()
            return delta

    def _train(self, itr):
        if itr % self.params.gp_train_freq == 0:
            self.optimizer.zero_grad()
            loss = self._compute_loss(*self.train_dataset)
            loss.backward()
            self.optimizer.step()

            if not self.is_trained:
                self.is_trained = True
            return loss.cpu().data.numpy()

    def _compute_loss(self, inputs, targets):
        return -self.mll_loss(self.model(inputs), targets)

    def _train_preproc(self, samples, stats):
        num_data = samples.rew.shape[0]
        holdout_ratio = self.params.holdout_ratio
        train_ids, valid_ids = train_valid_split(data_size=num_data, holdout_ratio=holdout_ratio)
        train_split, valid_split = self._samples2train_valid(samples=samples,
                                                             train_ids=train_ids,
                                                             valid_ids=valid_ids)

        self.train_dataset = self._samples2io(train_split, stats, is_normalized=self.params.normalized_io)
        # set training data
        self.model.set_train_data(*self.train_dataset, strict=False)

        self.valid_dataset = self._samples2io(valid_split, stats, is_normalized=self.params.normalized_io)

    @torch.no_grad()
    def eval(self):
        with gpytorch.settings.fast_pred_var():
            return self._compute_loss(*self.valid_dataset).cpu().data.numpy()


    def _instantiate_likelihood(self):
        if self.out_dim == 1:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        else:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.out_dim)
