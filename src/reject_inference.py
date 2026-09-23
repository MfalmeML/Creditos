import numpy as np
import pandas as pd


def infer_rejects(approved_pd, rejected_pd, inflation=1.3):
    # Simple upper-bound adjustment: rejected population assumed riskier
    return np.clip(rejected_pd * inflation, 0, 1)


if __name__ == '__main__':
    approved = np.array([0.02, 0.05, 0.10])
    rejected = np.array([0.15, 0.25, 0.40])
    print(infer_rejects(approved, rejected))
