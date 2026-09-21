import json
from pathlib import Path

def load_config(path='configs/settings.json'):
    return json.loads(Path(path).read_text())

if __name__ == '__main__':
    print(load_config())
