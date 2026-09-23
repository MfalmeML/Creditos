from src.optimizer import simulate_offers
from src.constraints import filter_offers


def pick_optimal(pd_, lgd, ead, max_ecl=2000.0, min_profit=0.0):
    offers = simulate_offers(pd_, lgd, ead)
    feasible = filter_offers(offers, max_ecl=max_ecl, min_profit=min_profit)
    if feasible.empty:
        return None
    return feasible.loc[feasible['profit'].idxmax()]


if __name__ == '__main__':
    print(pick_optimal(0.04, 0.6, 10000).to_dict())
