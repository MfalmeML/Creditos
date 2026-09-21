import numpy as np
from src.calibration import calibration_report

def test_calibration_runs():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7])
    r = calibration_report(y, p, n_bins=5)
    assert len(r) > 0
    print('ok')

if __name__ == '__main__':
    test_calibration_runs()
