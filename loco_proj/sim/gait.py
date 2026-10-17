import numpy as np
from dataclasses import dataclass
from swing_traj import *

@dataclass
class GaitConfig:
    ratio: float = 0.5
    period: float = 0.5
    rhythm: np.ndarray = np.array([[0, ratio, ratio, 0],
                                   [1, 0, 0, 1]])

def gait_scheduler(state: np.ndarray, cfg: GaitConfig) -> None:
    step_num = cfg.rhythm.shape[0]
    cumu_ratio = 0.
    for i in range(step_num):
        legs = np.nonzero(cfg.rhythm[i,:])
        len_temp = np.max(cfg.rhythm[i,:]) - cumu_ratio
        cumu_ratio += len_temp
        
    return