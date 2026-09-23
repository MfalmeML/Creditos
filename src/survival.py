import pandas as pd

def survival_at(model, row, t):
    df = pd.DataFrame([row])
    return float(model.predict_survival_function(df, times=[t]).iloc[0, 0])

def default_prob_within(model_default, model_prepay, row, t):
    d = survival_at(model_default, row, t)
    p = survival_at(model_prepay, row, t)
    # Combined: probability of default by t while still at risk
    return float((1 - d) * (1 - (1 - p) / 2))

if __name__ == '__main__':
    import joblib
    dm = joblib.load('models/default_hazard.joblib')
    pm = joblib.load('models/prepay_hazard.joblib')
    row = {'age': 40, 'credit_amount': 10000, 'rate': 0.15}
    print('PD 12m:', default_prob_within(dm, pm, row, 12))
