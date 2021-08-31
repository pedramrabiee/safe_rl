from trainers.base_trainer import BaseTrainer
from logger import logger

class DDPGTrainer(BaseTrainer):
    def _train(self, itr):
        # collect data by running current policy
        self.sampler.collect_data(itr)

        # prepare for training
        self.agent.train_mode(device=self.config.training_device)
        optim_dict = self._prep_optimizer_dict()
        for _ in range(self.agent.params.net_updates_per_iter):
            # get sample batch
            samples = self.sampler.sample(device=self.config.training_device)

            # train agent on sampled data
            optim_info = self.agent.optimize_agent(samples, optim_dict)

            # log training info
            logger.dump_tabular(cat_key='iteration', log=False, wandb_log=True, csv_log=False)

            # run after optimize
            self.agent.after_optimize()
