import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from src.drift import data_drift
from src.prediction_drift import prediction_drift


def model_risk_report(model_path='models/pd_baseline.joblib'):
    model = joblib.load(model_path)
    train_data = fetch_openml(name='credit-g', version=1, as_frame=True)
    ref_X = train_data.frame.drop(columns=['class'])

    cur = pd.read_csv('data/decisions_full.csv')
    cur_X = cur[ref_X.columns]

    dd = data_drift(ref_X, cur_X)
    pp = prediction_drift(
        model.predict_proba(ref_X)[:, 1],
        model.predict_proba(cur_X)[:, 1],
    )
    return {'data_drift_flagged': int(dd['drift'].sum()), 'prediction_drift': pp}


if __name__ == '__main__':
    print(model_risk_report())
