from filters.base_filter import BaseFilter
from networks.mlp import MLPNetwork
import numpy as np
import torch
from utils.grads import get_jacobian, get_grad
from utils.optim import qp_from_np
from utils.scale import action2newbounds, action2oldbounds
from utils import scale
from scipy.linalg import block_diag
from utils.torch_utils import row_wise_dot
from utils.misc import np_object2dict, torchify, hard_copy, polyak_update
from logger import logger
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from itertools import chain
from attrdict import AttrDict


class CBFFilter(BaseFilter):
    def initialize(self, params, init_dict=None):
        self.params = params

        # get the observation dim from observation process class
        self._obs_dim = self.obs_proc.obs_dim(proc_key='filter')

        # initialize filter network
        self.filter_net = MLPNetwork(in_dim=self._obs_dim, out_dim=1, **self.params.filter_net_kwargs)
        self.filter_optimizer = params.filter_optim_cls(self.filter_net.parameters(), **params.filter_optim_kwargs)

        self.models = [self.filter_net]
        self.optimizers = [self.filter_optimizer]
        self.models_dict = dict(filter_net=self.filter_net)
        self.optimizers_dict = dict(filter_optimizer=self.filter_optimizer)


    @torch.no_grad()
    def filter(self, obs, ac, filter_dict=None):
        # TODO: this method only works for single-obs single-ac (does not support mutliprocessing)
        # process observation to match the models' input requirement
        obs = self.obs_proc.proc(obs, proc_key='filter')

        ac_lim_high = scale.ac_old_bounds[1]
        ac = action2oldbounds(ac)

        obs_torch = torch.tensor(obs, dtype=torch.float32)
        mu_f, mu_g, std_f, std_g = filter_dict['dyn_bd'].pred_dyn
        with torch.enable_grad():
            dh_dx = get_jacobian(net=self.filter_net, x=obs_torch).detach().numpy()
        f_hat, g_hat = filter_dict['dyn_bd'].nom_dyn

        if not self.params.use_trained_dyn:
            mu_f = np.zeros_like(mu_f)
            mu_g = np.zeros_like(mu_g)
            std_f = np.zeros_like(std_f)
            std_g = np.zeros_like(std_g)

        cbf_value = self.filter_net(obs_torch).detach().numpy().squeeze()

        # return filtered action by solving the QP problem
        ac_max = np.ones_like(ac) * ac_lim_high

        h1 = np.vdot(dh_dx, f_hat + mu_f) -\
             self.params.k_delta * np.vdot(np.abs(dh_dx), std_f + (std_g * ac_max[:, np.newaxis]).sum(-1)) +\
             self.params.eta * cbf_value
        h1 = h1.astype('float64')   # cvxopt matrix method won't work with float32
        extra = (f_hat + mu_f) + ((g_hat + mu_g) * ac[:, np.newaxis]).sum(-1)

        g = -np.matmul(dh_dx, (g_hat + mu_g).squeeze(0))

        P = block_diag(2 * np.eye(self._ac_dim), self.params.k_epsilon)
        q = np.block([
            [-2 * ac.T],
            [0.0]
        ])

        G = np.block([
            [g, -1.0],
            [np.eye(self._ac_dim), np.zeros([self._ac_dim, 1])],
            [-np.eye(self._ac_dim), np.zeros([self._ac_dim, 1])]
        ])

        h = np.block([
            [h1],
            [ac_lim_high[:, np.newaxis] * np.ones([self._ac_dim, 1])],
            [ac_lim_high[:, np.newaxis] * np.ones([self._ac_dim, 1])]
        ])

        ac_filtered = qp_from_np(P=P, q=q, G=G, h=h)
        ac_filtered = ac_filtered[:-1]              # last element is epsilon

        # push plots
        logger.push_plot(data=cbf_value, plt_key='cbf_value')
        self.custom_plotter.filter_push_action((ac, ac_filtered))

        return action2newbounds(ac_filtered.T), extra


    def pre_train(self, samples, pre_train_dict=None):
        epoch = 0
        chained_keys = [['deriv_samples', 'dyn_safe']]
        train_gens = self._make_dataloader(samples, chained_keys=chained_keys)
        max_epoch = self.params.pretrain_max_epoch / self.params.pretrain_batch_to_sample_ratio
        pbar = tqdm(total=max_epoch, desc='Filter Pretraining Progress')
        while True:
            safe_samples = next(iter(train_gens['safe_samples']))[0]
            unsafe_samples = next(iter(train_gens['unsafe_samples']))[0]
            deriv_samples, dyn_safe = next(iter(train_gens['deriv_samples']))
            sample = AttrDict(safe_samples=safe_samples,
                              unsafe_samples=unsafe_samples,
                              deriv_samples=deriv_samples,
                              dyn_safe=dyn_safe)
            self.filter_optimizer.zero_grad()
            loss = self._compute_pretrain_loss(sample)
            loss.backward()
            self.filter_optimizer.step()

            logger.add_tabular({"Loss/CBF_Filter": loss.cpu().data.numpy()}, cat_key="cbf_epoch")
            logger.dump_tabular(cat_key="cbf_epoch", log=False, wandb_log=True, csv_log=False)

            epoch += 1
            pbar.update(1)
            if self._check_stop_criteria(samples) or epoch > max_epoch:
                pbar.close()
                break

    def optimize_agent(self, samples, optim_dict=None):
        # you need to break down the experience, to experience from safe set, unsafe set, and general experience
        # update safety filter on experience
        epoch = 0
        while True:
            # run one gradient descent
            self.filter_optimizer.zero_grad()
            loss = self._compute_loss(samples, optim_dict['itr'])
            loss.backward()
            self.filter_optimizer.step()
            logger.add_tabular({"Loss/CBF_Filter": loss.cpu().data.numpy()}, cat_key="cbf_epoch")
            logger.dump_tabular(cat_key="cbf_epoch", log=False, wandb_log=True, csv_log=False)
            epoch += 1
            if epoch >= self.params.max_epoch:
                break

    def _compute_pretrain_loss(self, samples=None):
        safe_samples = samples.safe_samples
        unsafe_samples = samples.unsafe_samples
        deriv_samples = samples.deriv_samples
        dyns = samples.dyn_safe

        # Safe loss
        safe_loss = torch.zeros(1, requires_grad=True)[0]
        unsafe_loss = torch.zeros(1, requires_grad=True)[0]
        deriv_loss = torch.zeros(1, requires_grad=True)[0]
        if not safe_samples.size(0) == 0:     # if the tensor is not empty
            safe_loss = (self._append_zeros(self.params.gamma_safe - self.filter_net(safe_samples))).max(dim=-1).values
            safe_loss = self._normalize_loss(safe_loss).mean()
        # Unsafe loss
        if not unsafe_samples.size(0) == 0:
            unsafe_loss = (self._append_zeros(self.params.gamma_unsafe + self.filter_net(unsafe_samples))).max(dim=-1).values
            unsafe_loss = self._normalize_loss(unsafe_loss).mean()
        # Derivative loss
        if not deriv_samples.size(0) == 0:
            deriv_loss = self._compute_deriv_loss(deriv_samples, dyns)
            deriv_loss = (self._append_zeros(self.params.gamma_safe - deriv_loss)).max(dim=-1).values
            deriv_loss = self._normalize_loss(deriv_loss).mean()

        # push loss plots, dumped in trainer
        logger.push_plot(np.stack([safe_loss.detach().numpy(),
                                   unsafe_loss.detach().numpy(),
                                   deriv_loss.detach().numpy()]).reshape(1, -1),
                         plt_key="loss_plots")

        return self.params.safe_loss_weight * safe_loss +\
               self.params.unsafe_loss_weight * unsafe_loss + \
               self.params.safe_deriv_loss_weight * deriv_loss

    def _compute_loss(self, samples, itr):
        obs = samples.obs
        next_obs = samples.next_obs
        dyn = samples.dyn_values
        ts = self._timestep

        if not obs.size(0) == 0:
            if self.params.train_on_jacobian:
                deriv_loss = self._compute_deriv_loss(obs, dyn)
            else:
                deriv_loss = (1/ts) * (self.filter_net(next_obs) + (self.params.eta * ts - 1) * self.filter_net(obs))
            loss = (self._append_zeros(self.params.gamma_dh - deriv_loss)).max(dim=-1).values
            loss = self.params.deriv_loss_weight * self._normalize_loss(loss).mean()
            loss += self._compute_pretrain_loss(samples)
            return loss
        return self._compute_pretrain_loss(samples)

    def _check_stop_criteria(self, samples):
        h_unsafe = self.filter_net(samples.unsafe_samples)
        h_dot_deriv_samples = self._compute_deriv_loss(samples.deriv_samples, samples.dyn_safe)
        if torch.max(h_unsafe) > self.params.stop_criteria_eps or torch.min(h_dot_deriv_samples) < self.params.stop_criteria_eps:
            return False
        return True

    @staticmethod
    def _append_zeros(x, dim=-1):
        return torch.cat((x, torch.zeros_like(x)), dim=dim)

    def _normalize_loss(self, loss):
        return torch.tanh(loss) if self.params.loss_tanh_normalization else loss

    def _compute_deriv_loss(self, obs, dyn):
        grad = get_grad(self.filter_net, obs, create_graph=True).squeeze(dim=0) # TODO: implement sqeeuze on one output in get_grad like in get_jacobian and remove squeeze from this
        deriv_loss = row_wise_dot(grad, torchify(dyn, device=obs.device))
        deriv_loss += self.params.eta * self.filter_net(obs)
        return deriv_loss

    def _make_dataloader(self, samples, chained_keys=None):
        all_chained_keys = list(chain(*chained_keys))
        ratio = self.params.pretrain_batch_to_sample_ratio
        min_batch_size = int(min([v.size(0) for _, v in samples.items()]) * ratio)
        make_data_gen = lambda x: DataLoader(dataset=TensorDataset(*x), batch_size=min_batch_size, shuffle=True)
        train_generator = {k: make_data_gen([v]) for k, v in samples.items() if k not in all_chained_keys}
        chained_generator = {item[0]: make_data_gen([samples[k] for k in item]) for item in chained_keys}
        return {**train_generator, **chained_generator}