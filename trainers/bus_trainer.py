from trainers.base_trainer import BaseTrainer
from attrdict import AttrDict
class BUSTrainer(BaseTrainer):
    def initialize(self):
        self.safe_set.late_initialize(init_dict=AttrDict(backup_agent=self.agent.shield))

    def _train(self, itr):
        self.sampler.collect_data(itr)
