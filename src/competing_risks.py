import pandas as pd
from lifelines import CoxPHFitter

def fit_competing_risks(df, event_code, features, duration_col='duration', event_col='event'):
    # Cause-specific hazard: treat other event types as censored
    d = df.copy()
    d['cause'] = (d[event_col] == event_code).astype(int)
    cph = CoxPHFitter()
    cph.fit(d[features + [duration_col, 'cause']],
            duration_col=duration_col, event_col='cause')
    return cph

if __name__ == '__main__':
    df = pd.read_csv('data/loan_events.csv')
    feats = ['age', 'credit_amount', 'rate']
    default_model = fit_competing_risks(df, event_code=1, features=feats)
    prepay_model = fit_competing_risks(df, event_code=2, features=feats)
    print('DEFAULT model:')
    print(default_model.summary[['coef', 'p']])
    print('PREPAY model:')
    print(prepay_model.summary[['coef', 'p']])
