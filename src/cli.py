import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMANDS = {
    'train-pd': ['python', '-m', 'src.train_pd'],
    'train-pd-reject': ['python', '-m', 'src.train_pd_reject_aware'],
    'ecl': ['python', '-m', 'src.pipeline_ecl'],
    'decide': ['python', '-m', 'src.run_full_decisions'],
    'survival': ['python', '-m', 'src.fit_hazards'],
    'monitor': ['python', '-m', 'src.monitor_accounts'],
    'uplift': ['python', '-m', 'src.uplift'],
    'intervene': ['python', '-m', 'src.intervention'],
    'fraud': ['python', '-m', 'src.fraud_credit_unified'],
    'stress': ['python', '-m', 'src.stress_report'],
    'governance': ['python', '-m', 'src.governance'],
    'audit': ['python', '-m', 'src.audit'],
    'serve': ['uvicorn', 'src.api:app', '--port', '8000'],
}


def main():
    p = argparse.ArgumentParser(prog='creditos')
    p.add_argument('command', choices=list(COMMANDS))
    args = p.parse_args()
    sys.exit(subprocess.call(COMMANDS[args.command], cwd=PROJECT_ROOT))


if __name__ == '__main__':
    main()
