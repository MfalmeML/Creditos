from src.config import load_config


def filter_offers(df, max_ecl=None, min_profit=None):
    cfg = load_config()
    max_ecl = cfg['risk_appetite_max_ecl'] if max_ecl is None else max_ecl
    min_profit = cfg['min_profit'] if min_profit is None else min_profit
    mask = (df['ecl'] <= max_ecl) & (df['profit'] >= min_profit)
    return df[mask].copy()


if __name__ == '__main__':
    from src.optimizer import simulate_offers
    print(filter_offers(simulate_offers(0.04, 0.6, 10000)))
