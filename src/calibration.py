import pandas as pd
from sklearn.calibration import calibration_curve


def calibration_report(y_true, y_prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    return pd.DataFrame({'mean_pred': mean_pred, 'frac_pos': frac_pos})


if __name__ == '__main__':
    import joblib
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    df = data.frame
    X = df.drop(columns=['class'])
    y = (df['class'] == 'bad').astype(int)
    model = joblib.load('models/pd_baseline.joblib')
    probs = model.predict_proba(X)[:, 1]
    print(calibration_report(y, probs).to_string())
