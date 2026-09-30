import pandas as pd

OFFERS = [
    {'limit': 5000,  'rate': 0.10, 'term': 12},
    {'limit': 10000, 'rate': 0.12, 'term': 24},
    {'limit': 20000, 'rate': 0.15, 'term': 36},
]


def simulate_offers(pd_, lgd, offers=OFFERS):
    """ECL and profit per candidate. EAD(candidate) = candidate limit,
    per docs/model_evaluation.md 'Optimizer fix — target behavior', item 1.
    """
    rows = []
    for o in offers:
        ead = o['limit']
        ecl = pd_ * lgd * ead
        revenue = o['limit'] * o['rate'] * (o['term'] / 12)
        profit = revenue - ecl
        rows.append({**o, 'ead': ead, 'ecl': ecl, 'revenue': revenue, 'profit': profit})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print(simulate_offers(0.04, 0.6))
