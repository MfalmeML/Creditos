import numpy as np
import pandas as pd


def estimate_lgd(n, seed=42):
    rng = np.random.default_rng(seed)
    lgd = rng.beta(a=2, b=5, size=n)
    return pd.Series(lgd, name='lgd')


if __name__ == '__main__':
    s = estimate_lgd(10)
    print(s.describe())
