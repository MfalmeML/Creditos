import pandas as pd
import joblib
from src.intervention import recommend_actions

def test_recommend_actions():
    model = joblib.load('models/uplift_tlearner.joblib')
    d = pd.DataFrame({
        'account_id': ['A1','A2'],
        'age': [30, 55],
        'credit_amount': [12000, 2000],
        'pd_horizon': [0.22, 0.02],
    })
    out = recommend_actions(d, model)
    assert 'action' in out.columns
    print('ok')

if __name__ == '__main__':
    test_recommend_actions()
