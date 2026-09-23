import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from src.augmented_training import build_augmented

def train_with_reject_inference(df, approved_mask):
    X_aug, y_aug = build_augmented(df, approved_mask)
    m = LogisticRegression(max_iter=1000)
    m.fit(X_aug, y_aug)
    return m, X_aug.columns

if __name__ == '__main__':
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    df = data.frame
    import numpy as np
    approved = np.random.default_rng(0).random(len(df)) < 0.7
    m, cols = train_with_reject_inference(df, approved)
    joblib.dump({'model': m, 'columns': list(cols)}, 'models/pd_reject_aware.joblib')
    print('saved', len(cols), 'features')
