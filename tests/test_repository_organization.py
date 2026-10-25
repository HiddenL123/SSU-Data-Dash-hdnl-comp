"""Migration checks without network access or historical model training."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/repository_organization'
MOVES = json.loads((ROOT / 'docs/repository_moves.json').read_text())
MODULES = [(a[:-3].replace('/', '.'), b[:-3].replace('/', '.'))
           for a, b in MOVES.items() if a.endswith('.py')]


@pytest.mark.parametrize('old,new', MODULES)
def test_canonical_and_legacy_modules_are_identical(old, new):
    canonical = importlib.import_module(new)
    assert importlib.import_module(old) is canonical
    for name, value in vars(canonical).items():
        if isinstance(value, type) and value.__module__ == new:
            # Protocol 0 GLOBAL opcodes exercise historical pickle paths directly.
            import pickle
            assert pickle.loads(f'c{old}\n{name}\n.'.encode()) is value


def test_features_import_first_without_loading_models():
    subprocess.run([sys.executable, '-c',
                    'import features; import sys; assert "models" not in sys.modules; '
                    'import models; import models.features; '
                    'assert features.make_features is models.features.make_features'],
                   cwd=ROOT, check=True, timeout=30)


@pytest.fixture(scope='module')
def before_migration():
    # Generated on origin/main before any file moves, with synthetic listings only.
    return joblib.load(FIXTURES / 'before_migration.joblib')


@pytest.mark.parametrize('index', range(24))
def test_old_fitted_providers_and_new_fits_match_exactly(before_migration, index, tmp_path):
    from features import make_features
    train, homes, records = before_migration
    name, backend, track, old_fitted, expected_fit, expected_transform = records[index]
    pd.testing.assert_frame_equal(old_fitted.transform(homes), expected_transform, check_exact=True)
    fresh = make_features(name, model=backend, track=track)
    pd.testing.assert_frame_equal(fresh.fit_transform(train), expected_fit, check_exact=True)
    pd.testing.assert_frame_equal(fresh.transform(homes), expected_transform, check_exact=True)
    path = tmp_path / 'provider.joblib'
    joblib.dump(fresh, path)
    pd.testing.assert_frame_equal(joblib.load(path).transform(homes), expected_transform, check_exact=True)


CLI_HELP = json.loads((FIXTURES / 'cli_help.json').read_text())


@pytest.mark.parametrize('script', CLI_HELP)
def test_cli_arguments_unchanged(script):
    result = subprocess.run([sys.executable, script, '--help'], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert result.stdout == CLI_HELP[script]
    if script in MOVES:
        canonical = MOVES[script][:-3].replace('/', '.')
        result = subprocess.run([sys.executable, '-m', canonical, '--help'], cwd=ROOT,
                                capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stderr
        assert result.stdout == CLI_HELP[script]


NOTEBOOK_STATES = json.loads((FIXTURES / 'notebook_state.json').read_text())


@pytest.mark.parametrize('old', NOTEBOOK_STATES)
def test_notebooks_preserve_state_and_resolve_local_imports(old):
    nbformat = pytest.importorskip('nbformat')
    from IPython.core.interactiveshell import InteractiveShell
    path = ROOT / MOVES.get(old, old)
    notebook = json.loads(path.read_text(encoding='utf-8'))
    nbformat.validate(nbformat.from_dict(notebook))
    shell = InteractiveShell.instance()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = shell.transform_cell(''.join(cell['source']))
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    continue
                for name in names:
                    top = name.split('.')[0]
                    if (ROOT / (top + '.py')).exists() or (ROOT / top / '__init__.py').exists():
                        module = importlib.import_module(name)
                        if isinstance(node, ast.ImportFrom):
                            for alias in node.names:
                                if alias.name != '*' and not hasattr(module, alias.name):
                                    importlib.import_module(name + '.' + alias.name)
        cell.pop('source')
    actual = hashlib.sha256(json.dumps(notebook, sort_keys=True).encode()).hexdigest()
    assert actual == NOTEBOOK_STATES[old]


@pytest.mark.parametrize('folder', ['.', 'experiments', 'experiments/legacy'])
def test_notebook_bootstrap_from_nested_working_directories(folder, monkeypatch):
    path = ROOT / 'experiments/interaction.ipynb'
    nb = json.loads(path.read_text(encoding='utf-8'))
    source = ''.join(nb['cells'][0]['source'])
    bootstrap = source.split('import matplotlib.pyplot')[0]
    monkeypatch.chdir(ROOT / folder)
    monkeypatch.delenv('DATA_DASH_ROOT', raising=False)
    namespace = {}
    exec(bootstrap, namespace)
    assert Path.cwd() == ROOT
    assert namespace['ROOT'] == ROOT


def test_notebook_bootstrap_explicit_root(tmp_path, monkeypatch):
    nb = json.loads((ROOT / 'experiments/interaction.ipynb').read_text(encoding='utf-8'))
    bootstrap = ''.join(nb['cells'][0]['source']).split('import matplotlib.pyplot')[0]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DATA_DASH_ROOT', str(ROOT))
    exec(bootstrap, {})
    assert Path.cwd() == ROOT


def test_saved_model_directory_does_not_move():
    from models.save_model import DEFAULT_SAVE_DIR
    assert DEFAULT_SAVE_DIR == ROOT / 'saved_models'


@pytest.mark.parametrize('name,index', [
    ('experiments/listaware_shotgun.ipynb', 2),
    ('experiments/realworld_avm_shotgun.ipynb', 2),
    ('ensemble_release.ipynb', 3),
    ('notebooks/quickstart.ipynb', 1),
])
def test_colab_aware_setup_works_locally_without_subprocesses(name, index, monkeypatch):
    monkeypatch.chdir(ROOT / 'experiments/legacy')
    monkeypatch.delenv('DATA_DASH_ROOT', raising=False)
    monkeypatch.delitem(sys.modules, 'google.colab', raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('Local setup must not clone or install packages')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    nb = json.loads((ROOT / name).read_text(encoding='utf-8'))
    source = ''.join(nb['cells'][index]['source'])
    from IPython.core.interactiveshell import InteractiveShell
    # Quickstart's autoreload extension does not participate in root discovery.
    source = source.split('%load_ext')[0]
    namespace = {}
    exec(InteractiveShell.instance().transform_cell(source), namespace)
    assert Path.cwd() == ROOT
    if name == 'ensemble_release.ipynb':
        for index in (4, 6):
            exec(''.join(nb['cells'][index]['source']), namespace)
