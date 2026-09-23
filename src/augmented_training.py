import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from src.reject_inference_v2 import fit_accept_model, parcel_rejects

def build_augmented(df, approved_mask, seed=0):
    X = df.drop(columns=['class'])
    y = (df['class'] == 'bad').astype(int)
    Xa = pd.get_dummies(X[approved_mask], drop_first=True)
    Xr = pd.get_dummies(X[~approved_mask], drop_first=True).reindex(columns=Xa.columns, fill_value=0)
    m = fit_accept_model(Xa, y[approved_mask])
    y_pseudo = parcel_rejects(Xr, m, k=5)
    X_aug = pd.concat([Xa, Xr], axis=0)
    y_aug = np.concatenate([y[approved_mask].values, y_pseudo])
    return X_aug, y_aug

if __name__ == '__main__':
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    df = data.frame
    rng = np.random.default_rng(0)
    approved = rng.random(len(df)) < 0.7
    X_aug, y_aug = build_augmented(df, approved)
    print(X_aug.shape, y_aug.mean())
