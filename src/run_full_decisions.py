import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from src.select_offer import pick_optimal
from src.explain import top_drivers
from src.counterfactual import counterfactual
from src.thin_file import is_thin_file

data = fetch_openml(name='credit-g', version=1, as_frame=True)
X = data.frame.drop(columns=['class'])
model = joblib.load('models/pd_baseline.joblib')
ecl_df = pd.read_csv('data/ecl_output.csv')

records = []
for i, row in X.iterrows():
    r = ecl_df.iloc[i]
    best = pick_optimal(r['pd'], r['lgd'], r['ead'])
    drivers = top_drivers(model, row.to_dict(), k=3)
    cf = counterfactual(model, row.to_dict(), 'credit_amount', row['credit_amount'] * 0.5)
    records.append({
        'pd': r['pd'], 'lgd': r['lgd'], 'ead': r['ead'], 'ecl': r['ecl'],
        'limit': best['limit'] if best is not None else None,
        'rate': best['rate'] if best is not None else None,
        'term': best['term'] if best is not None else None,
        'profit': best['profit'] if best is not None else None,
        'fallback': 'cashflow' if is_thin_file(row.to_dict()) else 'credit',
        'top_drivers': str(drivers),
        'counterfactual': str(cf),
    })

pd.DataFrame(records).to_csv('data/decisions_full.csv', index=False)
print('rows:', len(records))
