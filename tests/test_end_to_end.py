import subprocess
import sys

STEPS = [
    ['python', 'src/train_pd.py'],
    ['python', 'src/pipeline_ecl.py'],
    ['python', 'src/run_full_decisions.py'],
    ['python', 'src/stress_report.py'],
    ['python', 'src/governance.py'],
]

def test_end_to_end():
    for s in STEPS:
        r = subprocess.run(s, capture_output=True, text=True)
        assert r.returncode == 0, f'failed: {s}\n{r.stderr}'
    print('e2e ok')

if __name__ == '__main__':
    test_end_to_end()
