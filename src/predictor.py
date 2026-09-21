import pandas as pd
import joblib
import numpy as np
from src.lgd import estimate_lgd
from src.ead import estimate_ead
from src.ecl import expected_credit_loss
from src.select_offer import pick_optimal
from src.explain import top_drivers
from src.thin_file import is_thin_file

MODEL = joblib.load('models/pd_baseline.joblib')

def score(applicant: dict):
    x = pd.DataFrame([applicant])
    pd_hat = float(MODEL.predict_proba(x)[:, 1][0])
    lgd = float(estimate_lgd(1).iloc[0])
    ead = float(estimate_ead([applicant.get('credit_amount', 0)]).iloc[0])
    ecl = float(expected_credit_loss([pd_hat], [lgd], [ead]).iloc[0])
    best = pick_optimal(pd_hat, lgd, ead)
    drivers = top_drivers(MODEL, applicant, k=3)
    return {
        'pd': pd_hat, 'lgd': lgd, 'ead': ead, 'ecl': ecl,
        'limit': float(best['limit']) if best is not None else None,
        'rate': float(best['rate']) if best is not None else None,
        'term': int(best['term']) if best is not None else None,
        'profit': float(best['profit']) if best is not None else None,
        'fallback': 'cashflow' if is_thin_file(applicant) else 'credit',
        'top_drivers': drivers,
    }

if __name__ == '__main__':
    import json
    from pathlib import Path
    sample = json.loads(Path('configs/sample_applicant.json').read_text())
    print(score(sample))
