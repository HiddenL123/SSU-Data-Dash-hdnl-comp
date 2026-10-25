import hashlib
import json
from pathlib import Path
import subprocess

records = {}
for name in subprocess.check_output(['git', 'ls-files', '*.ipynb'], text=True).splitlines():
    notebook = json.loads(Path(name).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        cell.pop('source')
    records[name] = hashlib.sha256(json.dumps(notebook, sort_keys=True).encode()).hexdigest()
Path('tests/fixtures/repository_organization/notebook_state.json').write_text(json.dumps(records, indent=2)+'\n')
