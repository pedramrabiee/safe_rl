from trainers.base_trainer import BaseTrainer
from attrdict import AttrDict
from functools import partial

class BUSTrainer(BaseTrainer):
    def initialize(self):
        self.safe_set.late_initialize(init_dict=AttrDict(backup_agent=self.agent.shield))

    def _train(self, itr):

        self.custom_plotter.dump(itr=itr,
                                 dump_dict=dict(
                                     backup_set_funcs=self.agent.shield.get_backup_sets_for_contour(),
                                     safe_set_func=self.agent.shield.get_safe_set_for_contour(),
                                     viability_kernel_funcs=[partial(self.agent.shield.get_h_per_id_from_batch_of_obs,
                                                                     id=id)
                                                             for id in range(self.agent.shield.backup_set_size)]
                                 ))
        self.sampler.collect_data(itr)
