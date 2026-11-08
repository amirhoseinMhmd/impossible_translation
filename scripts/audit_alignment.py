"""Audit the arc-distance measurement.

1. LocalShuffle: enumerate every window order whose decoding reproduces the stored
   input, and record the range of each arc's perturbed distance across them.
2. All perturbations: record how many subwords each endpoint spans and the arc's
   perturbed distance under alternative anchors (nearest pieces, last pieces).
"""
import argparse
import itertools
import os
from pathlib import Path

import pandas as pd
import spacy

from arc_alignment import RECOVER, encode_with_offsets, tok

CAP = 512
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


def all_local_orders(actual, inp, window=3):
    text = ' ' + actual + ' '
    ids, offs = encode_with_offsets(text)
    target = inp.strip()
    windows = [list(range(i, min(i + window, len(ids)))) for i in range(0, len(ids), window)]
    sols, order = [], []

    def dfs(w):
        if len(sols) >= CAP:
            return
        if w == len(windows):
            if tok.decode([ids[i] for i in order]).strip() == target:
                sols.append(list(order))
            return
        for perm in itertools.permutations(windows[w]):
            order.extend(perm)
            prefix = tok.decode([ids[i] for i in order]).lstrip().rstrip('�').rstrip()
            if target.startswith(prefix):
                dfs(w + 1)
            del order[-len(perm):]

    dfs(0)
    return text, offs, sols


def pieces_of(doc_tokens, offs, shift):
    out = []
    for start, length in doc_tokens:
        s, e = start + shift, start + shift + length
        out.append([j for j, (a, b) in enumerate(offs) if b > s and a < e])
    return out


rows, sent_rows = [], []
for ds in ['bnc_spoken', 'gutenberg']:
    for pert in PERTS:
        d = pd.read_csv(Path(args.root) / ds / pert / '42' / f'full_samples_1k_{ds}_{pert}_final.csv')
        actual = [str(a) for a in d.actual]
        docs = list(nlp.pipe([a.strip() for a in actual], batch_size=64))
        for sid, (a, inp, doc) in enumerate(zip(actual, d.input.astype(str), docs)):
            if pert == 'localShuffle3':
                text, offs, sols = all_local_orders(a, inp)
                positions = [{o: n for n, o in enumerate(s)} for s in sols]
            else:
                text, _, offs, pos, ok = RECOVER[pert](a, inp)
                positions = [pos]
            shift = text.find(a.strip())
            pcs = pieces_of([(t.idx, len(t.text)) for t in doc], offs, shift)
            sent_rows.append(dict(dataset=ds, perturbation=pert, sent_id=sid, n_solutions=len(positions),
                                  capped=int(len(positions) >= CAP)))
            main = positions[0]
            for t in doc:
                if t.dep_ == 'ROOT':
                    continue
                dp, hp = pcs[t.i], pcs[t.head.i]
                firsts = [abs(p[dp[0]] - p[hp[0]]) for p in positions]
                rows.append(dict(dataset=ds, perturbation=pert, sent_id=sid, arc_idx=t.i,
                                 dist_first_min=min(firsts), dist_first_max=max(firsts),
                                 n_solutions=len(positions), capped=int(len(positions) >= CAP),
                                 dep_pieces=len(dp), head_pieces=len(hp),
                                 dist_nearest=min(abs(main[x] - main[y]) for x in dp for y in hp),
                                 dist_last=abs(main[dp[-1]] - main[hp[-1]])))
        print(ds, pert, len(rows), flush=True)

pd.DataFrame(rows).to_csv(Path(args.out) / 'alignment_audit_arcs.csv', index=False)
pd.DataFrame(sent_rows).to_csv(Path(args.out) / 'alignment_audit_sentences.csv', index=False)
print('done')
