import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def prediction_drift(ref_scores, cur_scores, alpha=0.05):
    stat, p = ks_2samp(ref_scores, cur_scores)
    return {'ks': float(stat), 'p': float(p), 'drift': bool(p < alpha)}


if __name__ == '__main__':
    rng = np.random.default_rng(1)
    print(prediction_drift(rng.normal(0.1, 0.05, 500), rng.normal(0.15, 0.05, 500)))
