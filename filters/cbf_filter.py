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
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


class CBFFilter(BaseFilter):
    def initialize(self, params, init_dict=None):
        self.params = params

        # get the observation dim from observation process class
        self._obs_dim = self.obs_proc.obs_dim(proc_key='filter')

        # initialize filter network
        self.filter_net = MLPNetwork(in_dim=self._obs_dim, out_dim=1, **self.params.filter_net_kwargs)
        self.filter_optimizer = params.filter_optim_cls(self.filter_net.parameters(), **params.filter_optim_kwargs)

        # self.filter_net_old = hard_copy(self.filter_net)

        self.models = [self.filter_net]
        self.optimizers = [self.filter_optimizer]
        self.models_dict = dict(filter_net=self.filter_net)
        self.optimizers_dict = dict(filter_optimizer=self.filter_optimizer)


    @torch.no_grad()
    def filter(self, obs, ac, filter_dict=None):
        # process observation to match the models' input requirement
        obs = self.obs_proc.proc(obs, proc_key='filter')

        # print(np.arctan2(obs[:, 1], obs[:, 0]))
        # TODO: get rid of scaler or clean it
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

        cbf_value = self.filter_net(obs_torch).detach().numpy().squeeze(0)

        # return filtered action by solving the QP problem
        # TODO: check the matmul for different ac_dim
        # TODO: self.params.eta * cbf_value has different dim than other term
        h1 = np.vdot(dh_dx, f_hat + mu_f) -\
             self.params.k_delta * np.vdot(np.abs(dh_dx), std_f + np.matmul(std_g, ac_lim_high[:, np.newaxis] * np.ones([self._ac_dim, 1])).squeeze(axis=-1)) +\
             self.params.eta * cbf_value

        #TODO: changed this from: extra = (f_hat + mu_f) + np.matmul((g_hat + mu_g).squeeze(0), ac).T check
        extra = (f_hat + mu_f) + np.matmul((g_hat + mu_g), ac.T).squeeze(-1)

        h1 = h1.astype('float64')   # cvxopt matrix method won't work with float32

        g = -np.matmul(dh_dx, (g_hat + mu_g).squeeze(axis=0))

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


        # extra_h = (f_hat + mu_f) + np.matmul((g_hat + mu_g).squeeze(axis=0), ac_filtered).T
        # logger.push_plot(data=cbf_value,
        #                  plt_key='cbf_value')
        #
        # logger.push_plot(np.concatenate((ac.reshape(1, -1), ac_filtered.reshape(1, -1)), axis=1), plt_key="sampler_plots")

        return action2newbounds(ac_filtered.T), extra


    def pre_train(self, samples, pre_train_dict=None):
        # TODO: fix observation processor
        self.safe_samples = torch.tensor(self.obs_proc.proc(samples['safe_samples'], proc_key='filter'), dtype=torch.float32)
        self.unsafe_samples = torch.tensor(self.obs_proc.proc(samples['unsafe_samples'], proc_key='filter'), dtype=torch.float32)
        self.deriv_samples = torch.tensor(self.obs_proc.proc(samples['deriv_samples'], proc_key='filter'), dtype=torch.float32)
        self.dyns = pre_train_dict['nom_dyn']

        epoch = 0
        while True:
            self.filter_optimizer.zero_grad()
            loss = self._compute_geometric_loss()
            loss.backward()
            self.filter_optimizer.step()

            logger.add_tabular({"Loss/CBF_Filter": loss.cpu().data.numpy()}, cat_key="cbf_epoch")
            epoch += 1
            if self._check_stop_criteria() or epoch > self.params.pretrain_max_epoch:
                break
        logger.dump_tabular(cat_key="cbf_epoch", log=False, wandb_log=True, csv_log=False)

    def optimize_agent(self, samples, optim_dict=None):
        # you need to break down the samples, to samples from safe set, unsafe set, and general samples
        # update safety filter on samples
        epoch = 0
        while True:
            # run one gradient descent
            self.filter_optimizer.zero_grad()
            loss = self._compute_loss(samples, optim_dict['itr'])
            loss.backward()
            self.filter_optimizer.step()
            logger.add_tabular({"Loss/CBF_Filter": loss.cpu().data.numpy()}, cat_key="cbf_epoch")
            epoch += 1
            # log
            if self._check_stop_criteria() or epoch >= self.params.max_epoch:
                break
        logger.dump_tabular(cat_key="cbf_epoch", log=False, wandb_log=True, csv_log=False)

    def _compute_geometric_loss(self, samples=None):
        # FIXME: if one of them gets samples as observation the other should get the observation. Why do you need
        #  anything other than the observation during training.

        if samples is None:
            safe_samples = self.safe_samples
            unsafe_samples = self.unsafe_samples
            deriv_samples = self.deriv_samples
            dyns = self.dyns
        else:
            safe_samples = samples.safe_samples.obs
            unsafe_samples = samples.unsafe_samples.obs
            deriv_samples = samples.deriv_samples.obs
            dyns = samples.dyns

        # Safe loss
        safe_loss = torch.zeros(1, requires_grad=True)[0]
        unsafe_loss = torch.zeros(1, requires_grad=True)[0]
        deriv_loss = torch.zeros(1, requires_grad=True)[0]
        if not safe_samples.size(0) == 0:     # if the tensor is not empty
            safe_loss = (self._append_zeros(self.params.gamma_safe - self.filter_net(safe_samples))).max(dim=-1).values.mean()
        # Unsafe loss
        if not unsafe_samples.size(0) == 0:
            unsafe_loss = (self._append_zeros(self.params.gamma_unsafe + self.filter_net(unsafe_samples))).max(dim=-1).values.mean()
        # Derivative loss 1
        if not deriv_samples.size(0) == 0:
            grad = get_grad(self.filter_net, deriv_samples, create_graph=True).squeeze(dim=0)  # TODO: implement sqeeuze on one output in get_grad like in get_jacobian and remove squeeze from this
            deriv_loss = row_wise_dot(grad, torchify(dyns, device=deriv_samples.device))
            deriv_loss += self.params.eta * self.filter_net(deriv_samples)
            deriv_loss = (self._append_zeros(self.params.gamma_safe - deriv_loss)).max(dim=-1).values.mean()

        # push loss plots
        logger.push_plot(np.stack([safe_loss.detach().numpy(),
                                   unsafe_loss.detach().numpy(),
                                   deriv_loss.detach().numpy()]).reshape(1, -1),
                         plt_key="loss_plots")

        return self.params.safe_loss_weight * safe_loss +\
               self.params.unsafe_loss_weight * unsafe_loss + \
               self.params.deriv_loss_weight * deriv_loss

    def _compute_loss(self, samples, itr):
        info = np_object2dict(samples.info)
        ts = self._timestep

        # TODO: Fix obs_proc
        next_obs = samples.next_obs
        obs = samples.obs
        if self.params.train_on_jacobian:
            grad = get_grad(self.filter_net, obs, create_graph=True).squeeze(
                dim=0)  # TODO: implement sqeeuze on one output in get_grad like in get_jacobian and remove squeeze from this
            deriv_loss = row_wise_dot(grad, torchify(info.dyn_out, device=obs.device))
            deriv_loss += self.params.eta * self.filter_net(obs)
        else:
            deriv_loss = (1/ts) * (self.filter_net(next_obs) + (self.params.eta * ts - 1) * self.filter_net(obs))
        loss = (self._append_zeros(self.params.gamma_dh - self.params.deriv_loss_weight * deriv_loss)).max(dim=-1).values.mean()

        loss += self._compute_geometric_loss(samples)
        return loss

    def _check_stop_criteria(self):
        h_unsafe = self.filter_net(self.unsafe_samples)
        h_safe = self.filter_net(self.safe_samples)
        # print(torch.max(h_unsafe))
        if torch.max(h_unsafe) > self.params.stop_criteria_eps:
        # or torch.min(h_safe) < -self.params.stop_criteria_eps:
            return False
        return True


    @staticmethod
    def _append_zeros(x, dim=-1):
        return torch.cat((x, torch.zeros_like(x)), dim=dim)

    def plotter(self, itr, max_speed):
        # FIXME: this only works for inverted pendulum env.
        speeds = max_speed * np.linspace(-1.0, 1.0, num=9)
        theta = np.linspace(-np.pi, np.pi, num=100).reshape(-1, 1)
        # plt.figure()
        for speed in speeds:
            x = np.concatenate((np.cos(theta), np.sin(theta), np.ones_like(theta) * speed), axis=-1)
            out = self.filter_net(torch.tensor(x, dtype=torch.float32)).detach().numpy()
            plt.plot(theta, out, label=r'$\dot \theta$ = ' + str(speed))
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$h$')
            plt.legend()

        logger.dump_plot(filename='cbf_itr_%d' % itr,
                         plt_key='cbf')

        # plt.figure()
        # mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer
        # plt.ion()
        speeds = max_speed * np.linspace(-1.0, 1.0, num=100)

        X, Y = np.meshgrid(theta, speeds)
        # x = np.concatenate((np.cos(X), np.sin(X), Y))

        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([np.cos(X[i, j]), np.sin(X[i, j]), Y[i, j]]).reshape(1,-1)
                out[i, j] = self.filter_net(torch.tensor(x, dtype=torch.float32)).detach().numpy().squeeze()

        ax = plt.axes(projection='3d')
        # ax.contour3D(X, Y, out, 50, cmap='binary')
        ax.plot_surface(X, Y, out, rstride=1, cstride=1,
                     cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot \theta$')
        ax.set_zlabel(r'$h$'),
        ax.view_init(50, 40)

        logger.dump_plot(filename='cbf_itr_%d_3D' % itr,
                         plt_key='cbf')

        # plt.ioff()


