import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def data_drift(ref, cur, cols=None, alpha=0.05):
    cols = cols or ref.columns
    rows = []
    for c in cols:
        stat, p = ks_2samp(ref[c].dropna(), cur[c].dropna())
        rows.append({'feature': c, 'ks': stat, 'p': p, 'drift': p < alpha})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({'x': rng.normal(0, 1, 500), 'y': rng.normal(0, 1, 500)})
    cur = pd.DataFrame({'x': rng.normal(0.5, 1, 500), 'y': rng.normal(0, 1, 500)})
    print(data_drift(ref, cur).to_string())
