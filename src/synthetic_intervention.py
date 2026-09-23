import numpy as np
import pandas as pd

def generate_intervention_data(n=8000, seed=42):
    rng = np.random.default_rng(seed)
    pd_base = rng.beta(2, 20, n)
    age = rng.integers(21, 65, n)
    amount = rng.integers(500, 20000, n)
    # Treatment assigned ~50% randomly (as in an experiment)
    treatment = rng.integers(0, 2, n)
    # Heterogeneous treatment effect: younger, larger loans respond better to restructuring
    te = np.where((age < 40) & (amount > 5000), -0.15, -0.03)
    pd_after = np.clip(pd_base + treatment * te, 0.001, 0.999)
    u = rng.random(n)
    default = (u < pd_after).astype(int)
    return pd.DataFrame({
        'age': age, 'credit_amount': amount,
        'treatment': treatment, 'default': default,
        'pd_baseline': pd_base,
    })

if __name__ == '__main__':
    df = generate_intervention_data()
    print(df.groupby('treatment')['default'].mean())
    df.to_csv('data/intervention.csv', index=False)
