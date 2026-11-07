"""Build per-arc and per-sentence tables for the RQ1-RQ3 analyses.

Arc positions in the perturbed input come from the exact GPT-2 token permutation
(arc_alignment.RECOVER), so lengths are in GPT-2 token units on both sides.
An arc counts as recovered when its lower-cased (dependent, relation, head)
triple occurs in the prediction's parse, matched as a multiset.
"""
import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

from arc_alignment import RECOVER, REVERSE_MARKER, tok, word_to_first_piece

PERTS = ['partialReverse', 'localShuffle3', 'fullShuffle']
SEEDS = [42, 1337, 2025, 314159, 8675309]

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
root, out = Path(args.root), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

try:
    spacy.require_gpu()
except Exception:
    pass
nlp = spacy.load(os.environ.get('SPACY_MODEL', 'en_core_web_trf'), disable=['ner'])


def parse(texts):
    return list(nlp.pipe([str(t) for t in texts], batch_size=64))


def triple(t):
    return (t.text.lower(), t.dep_, t.head.text.lower())


def adl(doc):
    ls = [abs(t.i - t.head.i) for t in doc if t.dep_ != 'ROOT']
    return float(np.mean(ls)) if ls else np.nan


def bigrams(ids):
    return set(zip(ids, ids[1:]))


arc_rows, sent_rows = [], []
for ds in ['gutenberg', 'bnc_spoken']:
    for pert in PERTS:
        frames = {s: pd.read_csv(root / ds / pert / str(s) / f'full_samples_1k_{ds}_{pert}_final.csv') for s in SEEDS}
        base = frames[SEEDS[0]]
        actual = [str(a) for a in base.actual]
        inputs = [str(i) for i in base.input]
        gold_docs = parse([a.strip() for a in actual])
        copy_docs = parse([i.replace(REVERSE_MARKER, ' ').strip() for i in inputs])

        gold_info = []
        for sid, (a, inp, gdoc) in enumerate(zip(actual, inputs, gold_docs)):
            text, ids, offs, pos, ok = RECOVER[pert](a, inp)
            assert ok, (ds, pert, sid)
            shift = text.find(a.strip())
            first = word_to_first_piece([(t.idx, len(t.text)) for t in gdoc], offs, shift)
            n_in = max(pos.values()) + 1
            in_ids = [None] * n_in
            for o, nidx in pos.items():
                in_ids[nidx] = ids[o]
            word_rank = {w: r for r, w in enumerate(sorted(range(len(gdoc)), key=lambda w: pos[first[w]]))}
            arcs = []
            for t in gdoc:
                if t.dep_ == 'ROOT':
                    continue
                d, h = first[t.i], first[t.head.i]
                wd, wh = word_rank[t.i], word_rank[t.head.i]
                arcs.append(dict(
                    arc_idx=t.i, relation=t.dep_, dep_pos_tag=t.pos_,
                    dep_lower=t.text.lower(), head_lower=t.head.text.lower(),
                    gold_arc_len_w=abs(t.i - t.head.i),
                    gold_arc_len_bpe=abs(d - h),
                    pert_arc_len_bpe=abs(pos[d] - pos[h]),
                    pert_arc_len_w=abs(wd - wh),
                    order_preserved=int((t.i < t.head.i) == (wd < wh)),
                    shared_piece=int(d == h),
                    triple=triple(t), dep_first=t.i < t.head.i))
            gold_info.append(dict(ids=ids, in_ids=[i for i in in_ids if i is not None], arcs=arcs, doc=gdoc))

        for seed, df in frames.items():
            assert (df.actual.astype(str).values == np.array(actual)).all()
            preds = [str(p) for p in df.prediction]
            pred_docs = parse(preds)
            for sid, (g, pdoc, pred) in enumerate(zip(gold_info, pred_docs, preds)):
                avail = Counter(triple(t) for t in pdoc if t.dep_ != 'ROOT')
                avail_ord = Counter((triple(t), t.i < t.head.i) for t in pdoc if t.dep_ != 'ROOT')
                n_rec = 0
                for arc in g['arcs']:
                    rec = int(avail[arc['triple']] > 0)
                    if rec:
                        avail[arc['triple']] -= 1
                    key = (arc['triple'], arc['dep_first'])
                    rec_ord = int(avail_ord[key] > 0)
                    if rec_ord:
                        avail_ord[key] -= 1
                    n_rec += rec
                    row = {k: v for k, v in arc.items() if k not in ('triple', 'dep_first')}
                    row.update(dataset=ds, perturbation=pert, seed=seed, sent_id=sid,
                               sent_len_w=len(g['doc']), sent_len_bpe=len(g['ids']), is_recovered=rec,
                               is_recovered_ordered=rec_ord)
                    arc_rows.append(row)
                pred_ids = tok.encode(pred.strip())
                in_bg = bigrams(g['in_ids'])
                pb, gb = bigrams(pred_ids), bigrams(g['ids'])
                arcs = g['arcs']
                sent_rows.append(dict(
                    dataset=ds, perturbation=pert, seed=seed, sent_id=sid,
                    sent_len_w=len(g['doc']), n_arcs=len(arcs), n_recovered=n_rec,
                    em=int(pred.strip().lower() == actual[sid].strip().lower()),
                    gold_adl_w=adl(g['doc']), pred_adl_w=adl(pdoc), copy_adl_w=adl(copy_docs[sid]),
                    gold_adl_bpe=np.mean([x['gold_arc_len_bpe'] for x in arcs]) if arcs else np.nan,
                    input_adl_bpe=np.mean([x['pert_arc_len_bpe'] for x in arcs]) if arcs else np.nan,
                    input_adl_w=np.mean([x['pert_arc_len_w'] for x in arcs]) if arcs else np.nan,
                    order_preserved=np.mean([x['order_preserved'] for x in arcs]) if arcs else np.nan,
                    copy_bigram=len(pb & in_bg) / len(pb) if pb else np.nan,
                    gold_bigram_in_input=len(gb & in_bg) / len(gb) if gb else np.nan))
            print(ds, pert, seed, len(arc_rows), flush=True)

pd.DataFrame(arc_rows).to_csv(out / 'arcs_v2.csv', index=False)
pd.DataFrame(sent_rows).to_csv(out / 'sentences_v2.csv', index=False)
print('done', len(arc_rows), len(sent_rows))
