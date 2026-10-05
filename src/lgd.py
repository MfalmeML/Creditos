import numpy as np
import pandas as pd

# Placeholder LGD for the served scoring path.
# This is NOT a learned LGD model and NOT applicant-derived.
# It equals the mean of the synthetic Beta(2, 5) draw used elsewhere,
# i.e. a / (a + b) = 2 / 7.
# Replace with a real LGD model when real recovery/collections data
# becomes available.
LGD_PLACEHOLDER = 2 / 7


def estimate_lgd(n, seed=42):
    """Synthetic LGD generator used by non-served pipelines (event
    simulation, stress scenarios). Do NOT use this in the served path;
    the served path uses LGD_PLACEHOLDER."""
    rng = np.random.default_rng(seed)
    lgd = rng.beta(a=2, b=5, size=n)
    return pd.Series(lgd, name='lgd')


if __name__ == '__main__':
    s = estimate_lgd(10)
    print(s.describe())
    print('placeholder:', LGD_PLACEHOLDER)
