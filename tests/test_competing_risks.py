import pandas as pd
from src.synthetic_events import generate_loan_events
from src.competing_risks import fit_competing_risks

def test_competing_risk_fit():
    df = generate_loan_events(n=500)
    cph = fit_competing_risks(df, event_code=1, features=['age', 'rate'])
    assert cph is not None
    print('ok')

if __name__ == '__main__':
    test_competing_risk_fit()
