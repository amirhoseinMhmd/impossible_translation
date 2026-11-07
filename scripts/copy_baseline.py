"""Copy-input Triple F1: score each perturbed input as if it were the model's output."""
import argparse
import os
from pathlib import Path

import pandas as pd
import spacy

REVERSE_MARKER = '\U0001f141'
PERTS = ['partialReverse', 'localShuffle3', 'fullShuffle']

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

try:
    spacy.require_gpu()
except Exception:
    pass
nlp = spacy.load(os.environ.get('SPACY_MODEL', 'en_core_web_trf'), disable=['ner'])


def triples(doc):
    return {(t.text.lower(), t.dep_, t.head.text.lower()) for t in doc if t.dep_ != 'ROOT'}


def f1(gold, pred):
    if not gold or not pred:
        return 0.0
    m = len(gold & pred)
    p, r = m / len(pred), m / len(gold)
    return 2 * p * r / (p + r) if p + r else 0.0


rows = []
for ds in ['bnc_spoken', 'gutenberg']:
    for pert in PERTS:
        d = pd.read_csv(Path(args.root) / ds / pert / '42' / f'full_samples_1k_{ds}_{pert}_final.csv')
        gold = list(nlp.pipe([str(a).strip() for a in d.actual], batch_size=64))
        inp = list(nlp.pipe([str(i).replace(REVERSE_MARKER, ' ').strip() for i in d.input], batch_size=64))
        for sid, (g, i) in enumerate(zip(gold, inp)):
            rows.append(dict(dataset=ds, perturbation=pert, sent_id=sid, copy_f1=f1(triples(g), triples(i))))
        print(ds, pert, flush=True)
out = pd.DataFrame(rows)
out.to_csv(Path(args.out) / 'copy_baseline_f1.csv', index=False)
print(out.groupby(['dataset', 'perturbation']).copy_f1.mean())
