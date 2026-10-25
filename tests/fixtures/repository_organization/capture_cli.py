import json
import subprocess
import sys
from pathlib import Path

results = {}
for p in Path('.').glob('*.py'):
    if 'argparse.ArgumentParser(' in p.read_text(encoding='utf-8'):
        run = subprocess.run([sys.executable, str(p), '--help'], capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, (p, run.stderr)
        results[str(p)] = run.stdout
Path('tests/fixtures/repository_organization/cli_help.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
print('Captured CLI help for', len(results), 'entry points')
