import numpy as np
import pandas as pd


def estimate_ead(balance, ccf=0.75, limit=None):
    balance = pd.Series(balance)
    if limit is None:
        return balance.rename('ead')
    undrawn = (pd.Series(limit) - balance).clip(lower=0)
    return (balance + ccf * undrawn).rename('ead')


if __name__ == '__main__':
    bal = [1000, 2000, 500]
    lim = [5000, 5000, 1000]
    print(estimate_ead(bal, limit=lim).tolist())
