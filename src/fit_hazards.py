import pandas as pd
import joblib
from src.competing_risks import fit_competing_risks

df = pd.read_csv('data/loan_events.csv')
feats = ['age', 'credit_amount', 'rate']
joblib.dump(fit_competing_risks(df, 1, feats), 'models/default_hazard.joblib')
joblib.dump(fit_competing_risks(df, 2, feats), 'models/prepay_hazard.joblib')
print('saved')
