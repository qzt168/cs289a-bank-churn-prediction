"""Local smoke harness; does not mount Drive or write into the repository's data folder.
Usage: python tests/validate_notebook.py --data /path/to/train.csv
Requires notebook packages, nbformat. Use --synthetic if the dataset is unavailable.
"""
import argparse
import ast
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import nbformat
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path)
parser.add_argument('--synthetic', action='store_true')
parser.add_argument('--notebook', type=Path, default=Path(__file__).resolve().parents[1] / 'bank_churn.ipynb')
args = parser.parse_args()
notebook = nbformat.read(args.notebook, as_version=4)
nbformat.validate(notebook)
for i, cell in enumerate(notebook.cells):
    if cell.cell_type == 'code':
        ast.parse(cell.source, filename=f'cell_{i}')
print('PASS JSON schema and all code syntax')
if not (args.data or args.synthetic):
    raise SystemExit('Supply --data or --synthetic for execution checks.')
with tempfile.TemporaryDirectory(prefix='churn_validation_') as temporary:
    root = Path(temporary)
    if args.synthetic:
        rng = np.random.default_rng(42)
        n = 1200
        synthetic = pd.DataFrame({
            'id': np.arange(n), 'CustomerId': np.arange(n), 'Surname': 'Example',
            'Geography': rng.choice(['France','Germany','Spain'], n),
            'Gender': rng.choice(['Female','Male'], n), 'Age': rng.integers(18,80,n),
            'CreditScore': rng.integers(300,850,n), 'Balance': rng.uniform(0,200000,n),
            'EstimatedSalary': rng.uniform(0,200000,n), 'Tenure': rng.integers(0,11,n),
            'NumOfProducts': rng.integers(1,5,n), 'HasCrCard': rng.integers(0,2,n),
            'IsActiveMember': rng.integers(0,2,n), 'Exited': rng.binomial(1,.25,n)})
        data_path = root / 'train.csv'
        synthetic.to_csv(data_path, index=False)
    else:
        data_path = args.data.resolve()
    namespace = {'__name__': '__notebook_test__'}
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        if 'from google.colab import drive' in cell.source:
            namespace.update(PROJECT_DIR=str(root), DATA_PATH=str(data_path), df=pd.read_csv(data_path))
            continue
        exec(compile(cell.source, f'cell_{i}', 'exec'), namespace)
    print('PASS top-to-bottom smoke execution (Drive loader substituted with local CSV)')
    assert not list((root/'results').glob('test_*'))
    assert not list((root/'results').glob('final_selection_*'))
    print('PASS test set remains unevaluated by default')
    table = pd.read_csv(root/'results/validation_results.csv')
    assert len(table) == 4 and set(table.model) == {'lr','svm','rf','mlp'}
    for config in namespace['REGISTRY'].values():
        first, path, loaded = namespace['train_or_load'](config)
        assert loaded
        _, before = namespace['evaluate'](first, namespace['val_idx'])
        # A cache hit must never call either fitting path.
        original_mlp = namespace['fit_mlp']
        original_estimator = namespace['make_estimator']
        def forbidden(*a, **k):
            raise AssertionError('Training unexpectedly called on checkpoint hit')
        namespace['fit_mlp'] = namespace['make_estimator'] = forbidden
        try:
            second, _, hit = namespace['train_or_load'](config)
            assert hit
        finally:
            namespace['fit_mlp'], namespace['make_estimator'] = original_mlp, original_estimator
        _, after = namespace['evaluate'](second, namespace['val_idx'])
        np.testing.assert_allclose(before, after, rtol=1e-7, atol=1e-8)
        changed = dict(config, implementation='changed-for-test')
        assert namespace['digest'](changed) != namespace['digest'](config)
        if config['model'] == 'mlp':
            history = first['history']
            assert first['best_epoch'] == min(history, key=lambda r:r['validation_loss'])['epoch']
    print('PASS all four checkpoint reloads skip fitting and reproduce predictions; best MLP epoch verified')
    # Retrain one changed configuration and verify it gets a distinct checkpoint.
    config = namespace['make_config']('lr', {'C': 0.123}, .10)
    _, changed_path, loaded = namespace['train_or_load'](config)
    assert not loaded and changed_path.exists()
    _, _, loaded = namespace['train_or_load'](config)
    assert loaded
    # Verify result upsert behavior without adding rows.
    row = table.iloc[0].to_dict()
    namespace['upsert_csv'](root/'results/validation_results.csv', row, ['experiment_id','mode','study'])
    assert len(pd.read_csv(root/'results/validation_results.csv')) == 4
    # Every configured feature / size / architecture produces a distinct identity.
    identity = namespace['digest']
    configs = [namespace['make_config']('lr', {'C':1.0}, f, group)
               for f in [.1,.25,.5,1.] for group in ['none',*namespace['GROUPS']]]
    assert len({identity(c) for c in configs}) == len(configs)
    assert namespace['select_configs']() == {}, 'Smoke results must not be final candidates'
    print('PASS changed hyperparameters retrain; result upsert; configuration identities; smoke selection guard')
    if args.synthetic:
        # Run final selection on inexpensive full-training synthetic/local configs in this temporary folder.
        namespace['MODE'] = 'full'
        for model, hp in namespace['DEFAULTS'].items():
            namespace['run_experiment'](model, hp, 'baseline', 1.0)
        namespace['RUN_FINAL_TEST'] = True
        final_cell = next(c.source for c in notebook.cells if c.cell_type == 'code' and 'def choose_threshold' in c.source)
        exec(final_cell, namespace)
        cached = pd.read_csv(root/'results/test_results.csv')
        assert len(cached) == 4
        # Cache reuse must not call test evaluation; threshold tuning remains disabled.
        original_evaluate = namespace['evaluate']
        namespace['evaluate'] = forbidden
        try:
            exec(final_cell, namespace)
        finally:
            namespace['evaluate'] = original_evaluate
        assert len(pd.read_csv(root/'results/test_results.csv')) == 4
        print('PASS final selection freeze, separate test persistence, and cached test reuse')
print('All checks passed. Temporary checkpoints and generated metrics were removed.')
