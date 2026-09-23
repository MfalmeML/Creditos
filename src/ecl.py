import pandas as pd


def expected_credit_loss(pd_, lgd, ead):
    pd_ = pd.Series(pd_)
    lgd = pd.Series(lgd)
    ead = pd.Series(ead)
    return (pd_ * lgd * ead).rename('ecl')


if __name__ == '__main__':
    out = expected_credit_loss([0.04, 0.10], [0.6, 0.5], [10000, 20000])
    print(out.tolist())
