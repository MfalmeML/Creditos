import json
from pathlib import Path
from src.predictor import score

base = json.loads(Path('configs/sample_applicant_homecredit.json').read_text())

low_pd = dict(base)
low_pd['EXT_SOURCE_1'] = 0.9
low_pd['EXT_SOURCE_2'] = 0.9
low_pd['EXT_SOURCE_3'] = 0.9

high_pd = dict(base)
high_pd['EXT_SOURCE_1'] = 0.05
high_pd['EXT_SOURCE_2'] = 0.05
high_pd['EXT_SOURCE_3'] = 0.05

results = {}
for label, applicant in [('low_pd', low_pd), ('high_pd', high_pd)]:
    r = score(applicant)
    results[label] = r
    print(label, 'pd=', round(r['pd'], 4),
          'limit=', r['limit'], 'term=', r['term'], 'profit=', round(r['profit'], 2))

print()
print('condition 1 (offers differ):', results['low_pd']['limit'] != results['high_pd']['limit'] or results['low_pd']['term'] != results['high_pd']['term'])
if results['low_pd']['limit'] is not None and results['high_pd']['limit'] is not None:
    print('condition 2 (direction: low_pd limit >= high_pd limit):', results['low_pd']['limit'] >= results['high_pd']['limit'])
else:
    print('condition 2: N/A, one side is None')
