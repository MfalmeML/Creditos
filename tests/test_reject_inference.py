import numpy as np
from src.synthetic_intervention import generate_intervention_data  # noqa
from sklearn.datasets import fetch_openml
from src.augmented_training import build_augmented

def test_augmented_shapes():
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    df = data.frame
    approved = np.random.default_rng(0).random(len(df)) < 0.7
    X, y = build_augmented(df, approved)
    assert len(X) == len(y)
    print('ok')

if __name__ == '__main__':
    test_augmented_shapes()
