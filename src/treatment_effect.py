import pandas as pd
import joblib
from src.uplift import TLearner


def treatment_effect(model, x_row):
    X = pd.DataFrame([x_row])
    p0 = model.m0.predict_proba(X)[:, 1][0]
    p1 = model.m1.predict_proba(X)[:, 1][0]
    return {
        'pd_no_intervention': float(p0),
        'pd_with_intervention': float(p1),
        'treatment_effect': float(p1 - p0),
    }

if __name__ == '__main__':
    model = joblib.load('models/uplift_tlearner.joblib')
    print(treatment_effect(model, {'age': 30, 'credit_amount': 12000}))
    print(treatment_effect(model, {'age': 55, 'credit_amount': 2000}))
