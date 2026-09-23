import random


def use_model(rollout_pct, rng=None):
    rng = rng or random
    return rng.random() < rollout_pct


if __name__ == '__main__':
    hits = sum(use_model(0.10) for _ in range(1000))
    print(f'model used ~{hits}/1000 at 10% rollout')
