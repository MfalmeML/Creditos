import pandas as pd
import joblib
from src.scenarios import apply_scenario, SCENARIOS
from src.survival import default_prob_within
from src.lgd import estimate_lgd
from src.ead import estimate_ead
from src.ecl import expected_credit_loss

def portfolio_ecl(portfolio_df, scenario_name,
                  default_model, prepay_model,
                  horizon_months=12):
    d = apply_scenario(portfolio_df, scenario_name)
    pd_hat = []
    for _, r in d.iterrows():
        pd_hat.append(default_prob_within(
            default_model, prepay_model,
            {'age': r['age'], 'credit_amount': r['credit_amount'], 'rate': r['rate']},
            horizon_months))
    pd_hat = pd.Series(pd_hat)
    lgd = estimate_lgd(len(d)).values
    ead = estimate_ead(d['credit_amount'].values).values
    ecl = expected_credit_loss(pd_hat, lgd, ead)
    return {
        'scenario': scenario_name,
        'n': len(d),
        'total_ecl': float(ecl.sum()),
        'mean_ecl': float(ecl.mean()),
        'mean_pd': float(pd_hat.mean()),
    }

if __name__ == '__main__':
    from src.synthetic_events import generate_loan_events
    dm = joblib.load('models/default_hazard.joblib')
    pm = joblib.load('models/prepay_hazard.joblib')
    port = generate_loan_events(n=2000)
    for name in SCENARIOS:
        print(portfolio_ecl(port, name, dm, pm))
