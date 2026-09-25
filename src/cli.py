import argparse
import subprocess
import sys

COMMANDS = {
    'train-pd': ['python', 'src/train_pd.py'],
    'train-pd-reject': ['python', 'src/train_pd_reject_aware.py'],
    'ecl': ['python', 'src/pipeline_ecl.py'],
    'decide': ['python', 'src/run_full_decisions.py'],
    'survival': ['python', 'src/fit_hazards.py'],
    'monitor': ['python', 'src/monitor_accounts.py'],
    'uplift': ['python', 'src/uplift.py'],
    'intervene': ['python', 'src/intervention.py'],
    'fraud': ['python', 'src/fraud_credit_unified.py'],
    'stress': ['python', 'src/stress_report.py'],
    'governance': ['python', 'src/governance.py'],
    'audit': ['python', 'src/audit.py'],
    'serve': ['uvicorn', 'src.api:app', '--port', '8000'],
}

def main():
    p = argparse.ArgumentParser(prog='creditos')
    p.add_argument('command', choices=list(COMMANDS))
    args = p.parse_args()
    sys.exit(subprocess.call(COMMANDS[args.command]))

if __name__ == '__main__':
    main()
