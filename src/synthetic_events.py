import numpy as np
import pandas as pd

def generate_loan_events(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    duration = rng.integers(3, 60, n)
    pd_base = rng.beta(2, 20, n)
    pp_hazard = rng.beta(2, 15, n)
    # Competing risks: event=1 default, event=2 prepay, event=0 censored
    u = rng.random(n)
    event = np.zeros(n, dtype=int)
    default_mask = u < pd_base
    prepay_mask = (~default_mask) & (u < pd_base + pp_hazard)
    event[default_mask] = 1
    event[prepay_mask] = 2
    return pd.DataFrame({
        'duration': duration,
        'event': event,
        'age': rng.integers(21, 65, n),
        'credit_amount': rng.integers(500, 20000, n),
        'rate': rng.uniform(0.05, 0.25, n),
    })

if __name__ == '__main__':
    df = generate_loan_events()
    print(df['event'].value_counts())
    df.to_csv('data/loan_events.csv', index=False)
