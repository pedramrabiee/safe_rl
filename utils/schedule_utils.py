import numpy as np


def capped_cubic_schedule(itr, capped_itr=1000):
    if itr < capped_itr:
        return int(round(itr ** (1. / 3))) ** 3 == itr
    else:
        return itr % capped_itr == 0

def multi_stage_schedule(itr, stages_dict):
    stages = np.array(stages_dict['stages'])
    stage_idx = np.where(stages - itr > 0)[0]
    stage_idx = len(stages) - 1 if stage_idx.size == 0 else stage_idx[0]

    return itr % stages_dict['freq'][stage_idx] == 0







