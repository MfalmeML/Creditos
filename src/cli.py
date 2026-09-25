import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMANDS = {
    'train-pd': [sys.executable, '-m', 'src.train_pd'],
    'train-pd-reject': [sys.executable, '-m', 'src.train_pd_reject_aware'],
    'ecl': [sys.executable, '-m', 'src.pipeline_ecl'],
    'decide': [sys.executable, '-m', 'src.run_full_decisions'],
    'survival': [sys.executable, '-m', 'src.fit_hazards'],
    'monitor': [sys.executable, '-m', 'src.monitor_accounts'],
    'uplift': [sys.executable, '-m', 'src.uplift'],
    'intervene': [sys.executable, '-m', 'src.intervention'],
    'fraud': [sys.executable, '-m', 'src.fraud_credit_unified'],
    'stress': [sys.executable, '-m', 'src.stress_report'],
    'governance': [sys.executable, '-m', 'src.governance'],
    'audit': [sys.executable, '-m', 'src.audit'],
    'serve': ['uvicorn', 'src.api:app', '--port', '8000'],
}


def main():
    p = argparse.ArgumentParser(prog='creditos')
    p.add_argument('command', choices=list(COMMANDS))
    args = p.parse_args()
    sys.exit(subprocess.call(COMMANDS[args.command], cwd=PROJECT_ROOT))


if __name__ == '__main__':
    main()
