import wandb
import numpy as np


def log_stat_wandb(stat, **kwargs):
    keys = list(stat.keys())
    for key in keys:
        try:
            if np.isnan(stat[key]):
                del stat[key]
        except Exception as e:
            continue

    wandb.log(stat, **kwargs)