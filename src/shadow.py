import json
from datetime import datetime, timezone
from pathlib import Path

SHADOW_LOG = Path('data/shadow_decisions.jsonl')

def log_shadow(applicant, model_decision, current_decision):
    SHADOW_LOG.parent.mkdir(exist_ok=True)
    rec = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'applicant': applicant,
        'model_decision': model_decision,
        'current_decision': current_decision,
    }
    with SHADOW_LOG.open('a') as f:
        f.write(json.dumps(rec) + '\n')

if __name__ == '__main__':
    log_shadow({'age': 35}, {'limit': 10000}, {'limit': 5000})
    print(SHADOW_LOG.read_text().splitlines()[-1])
