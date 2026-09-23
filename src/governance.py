import json
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from src.drift import data_drift
from src.prediction_drift import prediction_drift
from src.calibration import calibration_report
from src.fairness import approval_rate_parity
from src.model_risk_report import model_risk_report

def collect_governance():
    bundle = {}
    bundle['model_risk'] = model_risk_report()
    try:
        df = pd.read_csv('data/decisions_full.csv')
        bundle['n_decisions'] = int(len(df))
        bundle['approval_rate'] = float(df['limit'].notna().mean())
    except FileNotFoundError:
        bundle['n_decisions'] = 0
    try:
        stress = pd.read_csv('data/stress_report.csv')
        bundle['stress'] = stress.to_dict(orient='records')
    except FileNotFoundError:
        bundle['stress'] = []
    return bundle

if __name__ == '__main__':
    rep = collect_governance()
    with open('data/governance_report.json', 'w') as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2)[:600])
