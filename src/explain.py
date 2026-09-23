import numpy as np
import pandas as pd
import joblib


def top_drivers(model, x_row, k=5):
    prep = model.named_steps['prep']
    clf = model.named_steps['clf']
    x_t = prep.transform(pd.DataFrame([x_row]))
    names = prep.get_feature_names_out()
    coefs = clf.coef_[0]
    x_row_t = x_t.toarray()[0] if hasattr(x_t, "toarray") else np.asarray(x_t)[0]
    contrib = x_row_t * coefs
    idx = np.argsort(np.abs(contrib))[::-1][:k]
    return [(names[i], float(contrib[i])) for i in idx]


if __name__ == '__main__':
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    X = data.frame.drop(columns=['class'])
    model = joblib.load('models/pd_baseline.joblib')
    print(top_drivers(model, X.iloc[0].to_dict()))
