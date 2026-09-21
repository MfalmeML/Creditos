import pandas as pd

FORBIDDEN = ['class', 'default', 'outcome', 'recovery', 'settled']

def check_leakage(columns):
    hits = [c for c in columns if any(f in c.lower() for f in FORBIDDEN)]
    return hits

if __name__ == '__main__':
    cols = ['age', 'credit_amount', 'class', 'duration', 'recovery_rate']
    print(check_leakage(cols))
