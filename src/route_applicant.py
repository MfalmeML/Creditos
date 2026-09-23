import pandas as pd
import joblib

LABELS = {0: 'normal', 1: 'credit_distress', 2: 'fraud'}

def route(applicant):
    bundle = joblib.load('models/fraud_credit_router.joblib')
    m, feats = bundle['model'], bundle['features']
    X = pd.DataFrame([applicant])[feats]
    proba = m.predict_proba(X)[0]
    idx = int(proba.argmax())
    return {
        'route': LABELS[idx],
        'proba': {LABELS[i]: float(p) for i, p in enumerate(proba)},
    }

if __name__ == '__main__':
    print(route({'age': 30, 'credit_amount': 15000,
                 'device_reuse': 4, 'velocity': 8, 'geo_mismatch': 1}))
    print(route({'age': 45, 'credit_amount': 5000,
                 'device_reuse': 0, 'velocity': 1, 'geo_mismatch': 0}))
