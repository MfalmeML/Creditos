import pandas as pd
from src.select_offer import pick_optimal

df = pd.read_csv('data/ecl_output.csv')
rows = []
for _, r in df.iterrows():
    best = pick_optimal(r['pd'], r['lgd'], r['ead'])
    rows.append(best.to_dict() if best is not None else {'limit': None, 'rate': None, 'term': None, 'profit': None})

decisions = pd.DataFrame(rows)
decisions.to_csv('data/decisions.csv', index=False)
print(decisions.head())
print(decisions['limit'].value_counts(dropna=False))
