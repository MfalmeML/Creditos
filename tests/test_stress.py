import joblib
from src.synthetic_events import generate_loan_events
from src.portfolio_stress import portfolio_ecl

def test_stress_runs():
    dm = joblib.load('models/default_hazard.joblib')
    pm = joblib.load('models/prepay_hazard.joblib')
    port = generate_loan_events(n=300)
    r = portfolio_ecl(port, 'base', dm, pm)
    assert r['n'] == 300
    assert r['total_ecl'] >= 0
    print('ok')

if __name__ == '__main__':
    test_stress_runs()
