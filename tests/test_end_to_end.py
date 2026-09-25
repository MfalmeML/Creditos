import subprocess
import sys

STEPS = [
    [sys.executable, '-m', 'src.train_pd'],
    [sys.executable, '-m', 'src.pipeline_ecl'],
    [sys.executable, '-m', 'src.run_full_decisions'],
    [sys.executable, '-m', 'src.stress_report'],
    [sys.executable, '-m', 'src.governance'],
]


def test_end_to_end():
    for s in STEPS:
        r = subprocess.run(s, capture_output=True, text=True)
        assert r.returncode == 0, f'failed: {s}\n{r.stderr}'
    print('e2e ok')


if __name__ == '__main__':
    test_end_to_end()
