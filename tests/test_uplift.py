import pandas as pd
from src.synthetic_intervention import generate_intervention_data
from src.uplift import TLearner

def test_uplift_fit():
    df = generate_intervention_data(n=1000)
    m = TLearner().fit(df[['age','credit_amount']], df['treatment'], df['default'])
    u = m.predict_uplift(df[['age','credit_amount']])
    assert len(u) == len(df)
    print('ok')

if __name__ == '__main__':
    test_uplift_fit()
