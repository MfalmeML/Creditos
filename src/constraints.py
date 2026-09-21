import pandas as pd

def filter_offers(df, max_ecl=2000.0, min_profit=0.0):
    mask = (df['ecl'] <= max_ecl) & (df['profit'] >= min_profit)
    return df[mask].copy()

if __name__ == '__main__':
    from src.optimizer import simulate_offers
    offers = simulate_offers(0.04, 0.6, 10000)
    print(filter_offers(offers))
