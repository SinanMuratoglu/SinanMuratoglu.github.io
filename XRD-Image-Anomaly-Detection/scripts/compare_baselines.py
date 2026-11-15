"""Compare the full pipeline against random guessing and a brightness-stats-only baseline.

Usage: PYTHONPATH=. python3 scripts/compare_baselines.py DATA_ROOT
(DATA_ROOT holds data/features.csv, data/labels.csv and labels_{train,val,test}.csv)
Writes comparison.csv; plot it with scripts/make_comparison.py comparison.csv OUT.png
"""
import sys, numpy as np, pandas as pd
from xrdpipe.io import CLASSES, load_labels, load_split_ids
from xrdpipe.model import make_model, _proba
from sklearn.metrics import average_precision_score, roc_auc_score
D = sys.argv[1]
F = pd.read_csv(f'{D}/data/features.csv', index_col=0); F.index = F.index.astype(str)
L = load_labels(f'{D}/data/labels.csv')
sp = {k: load_split_ids(f'{D}/labels_{k}.csv') for k in ('train', 'val', 'test')}
data = F.join(L, how='inner'); tr, te = data.loc[sp['train']], data.loc[sp['test']]
sets = {'Simple brightness stats': ['int_mean', 'int_std', 'int_p50', 'int_p99'], 'Full pipeline': list(F.columns)}
rows = []
y = te[CLASSES].values
for name, cols in sets.items():
    p = _proba(make_model().fit(tr[cols], tr[CLASSES].values), te[cols])
    for j, c in enumerate(CLASSES):
        rows.append({'model': name, 'class': c, 'ap': average_precision_score(y[:, j], p[:, j]), 'auc': roc_auc_score(y[:, j], p[:, j])})
    np.save(f'proba_{name.split()[0]}.npy', p)
for j, c in enumerate(CLASSES):
    rows.append({'model': 'Random guessing', 'class': c, 'ap': y[:, j].mean(), 'auc': 0.5})
R = pd.DataFrame(rows); R.to_csv('comparison.csv', index=False)
np.save('ytest.npy', y)
print(R.pivot(index='class', columns='model', values='ap').round(3))
print(R.groupby('model')[['ap', 'auc']].mean().round(3))
