import numpy as np
import pandas as pd
import joblib

def fit_accept_model(X_accept, y_accept):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=1000)
    m.fit(X_accept, y_accept)
    return m

def parcel_rejects(X_reject, accept_model, k=5):
    """Parcelling: split rejects into risk-ordered bins, assign each bin
    the average bad rate of a comparable accepted bin (shifted conservatively)."""
    p = accept_model.predict_proba(X_reject)[:, 1]
    order = np.argsort(p)
    bins = np.array_split(order, k)
    accepted_p = accept_model.predict_proba(X_reject)[:, 1]
    y_pseudo = np.zeros(len(X_reject), dtype=int)
    for i, b in enumerate(bins):
        # Upper bins assumed riskier than accepted; assign 1 for top bins
        rate = min(0.05 + 0.15 * (i + 1), 0.9)
        n_bad = int(round(rate * len(b)))
        y_pseudo[b[:n_bad]] = 1
    return y_pseudo

if __name__ == '__main__':
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    df = data.frame
    X = df.drop(columns=['class'])
    y = (df['class'] == 'bad').astype(int)
    # Simulate: top 30% by a naive score get "approved"
    naive = pd.Series(np.random.default_rng(0).random(len(X)), index=X.index)
    approved = naive < 0.7
    Xa = pd.get_dummies(X[approved], drop_first=True)
    Xr = pd.get_dummies(X[~approved], drop_first=True).reindex(columns=Xa.columns, fill_value=0)
    m = fit_accept_model(Xa, y[approved])
    y_pseudo = parcel_rejects(Xr, m, k=5)
    print('approved bad rate:', y[approved].mean())
    print('reject pseudo bad rate:', y_pseudo.mean())
