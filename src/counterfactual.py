import pandas as pd
import joblib

def counterfactual(model, x_row, feature, new_value):
    x2 = dict(x_row)
    x2[feature] = new_value
    p1 = model.predict_proba(pd.DataFrame([x_row]))[:, 1][0]
    p2 = model.predict_proba(pd.DataFrame([x2]))[:, 1][0]
    return {'feature': feature, 'from': x_row[feature], 'to': new_value,
            'pd_before': float(p1), 'pd_after': float(p2),
            'delta': float(p2 - p1)}

if __name__ == '__main__':
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    X = data.frame.drop(columns=['class'])
    model = joblib.load('models/pd_baseline.joblib')
    row = X.iloc[0].to_dict()
    print(counterfactual(model, row, 'credit_amount', row['credit_amount'] * 0.5))
