import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def generate_unified_data(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 65, n)
    amount = rng.integers(500, 20000, n)
    # Fraud signals: device reuse, velocity, unusual geo
    device_reuse = rng.integers(0, 5, n)
    velocity = rng.integers(0, 10, n)
    geo_mismatch = rng.integers(0, 2, n)
    # True label: 0 normal, 1 credit distress, 2 fraud
    distress = (amount > 12000) & (rng.random(n) < 0.4)
    fraud = (device_reuse >= 3) & (velocity >= 6) | (geo_mismatch == 1) & (rng.random(n) < 0.5)
    label = np.zeros(n, dtype=int)
    label[distress] = 1
    label[fraud] = 2
    return pd.DataFrame({
        'age': age, 'credit_amount': amount,
        'device_reuse': device_reuse, 'velocity': velocity,
        'geo_mismatch': geo_mismatch, 'label': label,
    })

def train_router(df):
    feats = ['age', 'credit_amount', 'device_reuse', 'velocity', 'geo_mismatch']
    m = GradientBoostingClassifier(n_estimators=150, max_depth=3)
    m.fit(df[feats], df['label'])
    return m, feats

if __name__ == '__main__':
    df = generate_unified_data()
    print(df['label'].value_counts())
    m, feats = train_router(df)
    import joblib
    joblib.dump({'model': m, 'features': feats}, 'models/fraud_credit_router.joblib')
    df.to_csv('data/unified_fraud_credit.csv', index=False)
