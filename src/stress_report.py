import pandas as pd
import joblib
from src.synthetic_events import generate_loan_events
from src.scenarios import SCENARIOS
from src.portfolio_stress import portfolio_ecl

def run_all_scenarios(portfolio_df, dm, pm):
    rows = [portfolio_ecl(portfolio_df, s, dm, pm) for s in SCENARIOS]
    return pd.DataFrame(rows)

if __name__ == '__main__':
    dm = joblib.load('models/default_hazard.joblib')
    pm = joblib.load('models/prepay_hazard.joblib')
    port = generate_loan_events(n=2000)
    out = run_all_scenarios(port, dm, pm)
    out.to_csv('data/stress_report.csv', index=False)
    print(out.to_string())
