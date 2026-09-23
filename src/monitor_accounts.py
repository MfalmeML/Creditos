import pandas as pd
import joblib
from src.survival import default_prob_within

def rescore_accounts(accounts_df, default_model, prepay_model, horizon_months=12):
    rows = []
    for _, r in accounts_df.iterrows():
        pd_h = default_prob_within(default_model, prepay_model, r.to_dict(), horizon_months)
        rows.append({'account_id': r.get('account_id', None), 'pd_horizon': pd_h})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    dm = joblib.load('models/default_hazard.joblib')
    pm = joblib.load('models/prepay_hazard.joblib')
    accounts = pd.DataFrame({
        'account_id': ['A1', 'A2', 'A3'],
        'age': [35, 50, 28],
        'credit_amount': [8000, 15000, 3000],
        'rate': [0.12, 0.10, 0.20],
    })
    print(rescore_accounts(accounts, dm, pm).to_string())
