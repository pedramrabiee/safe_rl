from trainers.base_trainer import BaseTrainer
from logger import logger
from dynamics.gaussian_processes_dynamics import GPDynamics


class MBTrainer(BaseTrainer):
    def _train(self, itr):
        start_second_ctrl_itr = 1

        if itr == 0 and self.config.load_models and isinstance(self.agent.dynamics, GPDynamics) and self.agent.buffer_size() is not None:
            # Pretrain GP model on loaded data
            # FIXME: fix this if you found a way to load GP models and get inference from it without training
            buffer_data = self.agent.get_buffer(to_tensor=True, device=self.config.training_device) # load the entire buffer
            # train agent on sampled data
            self.agent.train_mode(device=self.config.training_device)

            optim_dict = self._prep_optimizer_dict()
            optim_dict['itr'] = itr
            optim_info = self.agent.optimize_agent(buffer_data, optim_dict)

            start_second_ctrl_itr = 2

        if itr == start_second_ctrl_itr:
            self.agent.init_controller(idx=1)

        # collect data by running current policy
        self.sampler.collect_data(itr)

        # get sample batch
        samples = self.sampler.sample(device=self.config.training_device)

        # train agent on sampled data
        self.agent.train_mode(device=self.config.training_device)

        optim_dict = self._prep_optimizer_dict()
        optim_dict['itr'] = itr
        optim_info = self.agent.optimize_agent(samples, optim_dict)

        logger.dump_tabular(cat_key='iteration', log=False, wandb_log=True, csv_log=False)

        # run after optimize
        self.agent.after_optimize()
