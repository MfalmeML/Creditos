import numpy as np
import pandas as pd

SCENARIOS = {
    'base':         {'unemployment_delta': 0.00, 'income_delta': 0.00, 'rate_delta': 0.00},
    'mild_stress':  {'unemployment_delta': 0.02, 'income_delta': -0.03, 'rate_delta': 0.01},
    'severe_stress':{'unemployment_delta': 0.05, 'income_delta': -0.08, 'rate_delta': 0.03},
    'combined_shock':{'unemployment_delta': 0.07, 'income_delta': -0.12, 'rate_delta': 0.05},
}

def apply_scenario(df, scenario_name):
    s = SCENARIOS[scenario_name]
    d = df.copy()
    d['age'] = d['age']
    d['credit_amount'] = d['credit_amount'] * (1 + s['income_delta'])
    d['rate'] = d['rate'] + s['rate_delta']
    d['scenario'] = scenario_name
    return d

if __name__ == '__main__':
    from src.synthetic_events import generate_loan_events
    df = generate_loan_events(n=2000)
    for name in SCENARIOS:
        print(name, apply_scenario(df, name)['credit_amount'].mean())
