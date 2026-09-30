import json
import joblib
from pathlib import Path

model = joblib.load('models/pd_baseline.joblib')
expected = set(model.feature_names_in_)
sample = json.loads(Path('configs/sample_applicant_homecredit.json').read_text())
missing = expected - set(sample.keys())
extra = set(sample.keys()) - expected
print('expected:', len(expected))
print('missing:', sorted(missing))
print('extra:', sorted(extra))
