import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from src.lgd import estimate_lgd
from src.ead import estimate_ead
from src.ecl import expected_credit_loss

data = fetch_openml(name='credit-g', version=1, as_frame=True)
df = data.frame
X = df.drop(columns=['class'])

model = joblib.load('models/pd_baseline.joblib')
pd_hat = model.predict_proba(X)[:, 1]

n = len(X)
lgd = estimate_lgd(n)
ead = estimate_ead(df['credit_amount'].values)

out = pd.DataFrame({
    'pd': pd_hat,
    'lgd': lgd.values,
    'ead': ead.values,
})
out['ecl'] = expected_credit_loss(out['pd'], out['lgd'], out['ead'])
print(out.head())
print(out['ecl'].describe())
out.to_csv('data/ecl_output.csv', index=False)
