"""RQ1-RQ3 analyses on arcs_v2.csv / sentences_v2.csv (output of build_arc_data.py).

All SEs are cluster-robust by source sentence (dataset:sent_id), which pools the
5 training seeds and 3 perturbations of the same sentence into one cluster.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'evaluation_results' / 'rq_v2'
OUT = IN
PERTS = ['partialReverse', 'localShuffle3', 'fullShuffle']
MIN_REL = int(sys.argv[2]) if len(sys.argv) > 2 else 200

arcs = pd.read_csv(IN / 'arcs_v2.csv')
sents = pd.read_csv(IN / 'sentences_v2.csv')
for d in (arcs, sents):
    d['item'] = d.dataset + ':' + d.sent_id.astype(str)
    d['perturbation'] = pd.Categorical(d.perturbation, PERTS)


def word_freq(ds):
    c = Counter()
    with open(ROOT / 'datasets' / f'100k_{ds}.train', encoding='utf-8', errors='ignore') as f:
        for line in f:
            c.update(re.findall(r"\w+|[^\w\s]", line.lower()))
    return c


arcs['log_dist'] = np.log2(arcs.pert_arc_len_bpe.clip(lower=1))
arcs['log_gold'] = np.log2(arcs.gold_arc_len_bpe.clip(lower=1))
for ds in arcs.dataset.unique():
    m = arcs.dataset == ds
    wf = word_freq(ds)
    arcs.loc[m, 'log_wfreq'] = np.log10(arcs.loc[m, 'dep_lower'].map(lambda w: wf.get(str(w), 0)) + 1)
    rel_counts = arcs[m & (arcs.seed == arcs.seed.min())].groupby('relation').size()
    arcs.loc[m, 'log_rfreq'] = np.log10(arcs.loc[m, 'relation'].map(rel_counts))
    arcs.loc[m, 'sent_len_z'] = (arcs.loc[m, 'sent_len_bpe'] - arcs.loc[m, 'sent_len_bpe'].mean()) / arcs.loc[m, 'sent_len_bpe'].std()


def fit_logit(formula, d):
    return smf.logit(formula, data=d).fit(disp=0, maxiter=200, cov_type='cluster',
                                          cov_kwds={'groups': pd.factorize(d['item'])[0]})


def wald(model, names):
    r = np.zeros((len(names), len(model.params)))
    for i, n in enumerate(names):
        r[i, list(model.params.index).index(n)] = 1
    t = model.wald_test(r, scalar=True)
    return float(np.squeeze(t.statistic)), len(names), float(t.pvalue)


def tidy(fit, **extra):
    ci = fit.conf_int()
    return pd.DataFrame(dict(term=fit.params.index, b=fit.params.values, se=fit.bse.values,
                             p=fit.pvalues.values, OR=np.exp(fit.params.values),
                             OR_lo=np.exp(ci[0].values), OR_hi=np.exp(ci[1].values), **extra))


def run_rq1(d, ds, y, rq1_rows, slope_rows):
    base = f'{y} ~ log_dist + log_gold + sent_len_z'
    m0 = fit_logit(base, d)
    m1 = fit_logit(base + ' + C(perturbation)', d)
    m2 = fit_logit(f'{y} ~ log_dist * C(perturbation) + log_gold + sent_len_z', d)
    pert_terms = [t for t in m1.params.index if t.startswith('C(perturbation)')]
    int_terms = [t for t in m2.params.index if t.startswith('log_dist:')]
    w_pert, w_int = wald(m1, pert_terms), wald(m2, int_terms)
    for name, m in [('M0', m0), ('M1', m1), ('M2', m2)]:
        rq1_rows.append(tidy(m, dataset=ds, model=name, llf=m.llf, aic=m.aic, mcfadden=m.prsquared))
    rq1_rows.append(pd.DataFrame([dict(dataset=ds, model='wald', term='perturbation (M1)', b=w_pert[0], se=w_pert[1], p=w_pert[2]),
                                  dict(dataset=ds, model='wald', term='log_dist x perturbation (M2)', b=w_int[0], se=w_int[1], p=w_int[2])]))
    cov = m2.cov_params()
    for p in PERTS:
        names = ['log_dist'] + ([f'log_dist:C(perturbation)[T.{p}]'] if p != PERTS[0] else [])
        b = sum(m2.params[n] for n in names)
        se = np.sqrt(sum(cov.loc[a, c] for a in names for c in names))
        dp = d[d.perturbation == p]
        sep = fit_logit(base, dp)
        slope_rows.append(dict(dataset=ds, perturbation=p, slope=b, se=se, lo=b - 1.96 * se, hi=b + 1.96 * se,
                               sep_slope=sep.params['log_dist'], sep_lo=sep.conf_int().loc['log_dist', 0],
                               sep_hi=sep.conf_int().loc['log_dist', 1], recovery=dp[y].mean()))


def vif(d, cols):
    x = d[cols].to_numpy(dtype=float)
    out = {}
    for i, c in enumerate(cols):
        others = np.column_stack([np.ones(len(x)), np.delete(x, i, axis=1)])
        resid = x[:, i] - others @ np.linalg.lstsq(others, x[:, i], rcond=None)[0]
        r2 = 1 - resid.var() / x[:, i].var()
        out[f'vif_{c}'] = 1 / (1 - r2)
    return out


# GPT-2 splits "didn't" as "didn"+"'t", so a clitic negator is tied to its auxiliary's subword.
arcs['relation'] = np.where((arcs.relation == 'neg') & arcs.dep_lower.isin(["n't", "n’t"]), 'neg_clitic', arcs.relation)

HAS_ORD = 'is_recovered_ordered' in arcs.columns
rq1_rows, slope_rows, rq1o_rows, slopeo_rows, rq2_rows, rq2_tests, desc_rows, vif_rows = [], [], [], [], [], [], [], []
for ds, d in arcs.groupby('dataset'):
    d = d.copy()
    agg = dict(n_arcs=('is_recovered', 'size'), recovery=('is_recovered', 'mean'),
               median_pert_len=('pert_arc_len_bpe', 'median'), median_gold_len=('gold_arc_len_bpe', 'median'))
    if HAS_ORD:
        agg['recovery_ordered'] = ('is_recovered_ordered', 'mean')
    desc_rows.append(d.groupby('perturbation', observed=True).agg(**agg).reset_index().assign(dataset=ds))

    run_rq1(d, ds, 'is_recovered', rq1_rows, slope_rows)
    if HAS_ORD:
        run_rq1(d, ds, 'is_recovered_ordered', rq1o_rows, slopeo_rows)
    for p, dp in d.groupby('perturbation', observed=True):
        vif_rows.append(dict(dataset=ds, perturbation=p, r_dist_sentlen=np.corrcoef(dp.log_dist, dp.sent_len_z)[0, 1],
                             r_dist_gold=np.corrcoef(dp.log_dist, dp.log_gold)[0, 1],
                             **vif(dp, ['log_dist', 'log_gold', 'sent_len_z'])))

    keep = d.relation.value_counts()
    keep = keep[keep >= MIN_REL * 5].index
    dr = d[d.relation.isin(keep)].copy()
    ctrl = 'is_recovered ~ log_dist + log_gold + sent_len_z + C(perturbation)'
    r0 = fit_logit(ctrl, dr)
    r1 = fit_logit(ctrl + ' + log_rfreq + log_wfreq', dr)
    r2 = fit_logit(ctrl + " + log_wfreq + C(relation, Treatment('nsubj'))", dr)
    rel_terms = [t for t in r2.params.index if t.startswith('C(relation')]
    w_rel = wald(r2, rel_terms)
    rq2_tests.append(dict(dataset=ds, n_arcs=len(dr), n_relations=len(keep),
                          b_rfreq=r1.params['log_rfreq'], p_rfreq=r1.pvalues['log_rfreq'],
                          b_wfreq=r1.params['log_wfreq'], p_wfreq=r1.pvalues['log_wfreq'],
                          wald_relation=w_rel[0], df_relation=w_rel[1], p_relation=w_rel[2],
                          mcf_ctrl=r0.prsquared, mcf_freq=r1.prsquared, mcf_rel=r2.prsquared))
    t = tidy(r2, dataset=ds)
    t = t[t.term.str.startswith('C(relation')].copy()
    t['relation'] = t.term.str.extract(r"T\.([^\]]+)\]")[0]
    stats = dr[dr.seed == dr.seed.min()].groupby('relation').agg(n=('is_recovered', 'size'), mean_gold_len=('gold_arc_len_w', 'mean'))
    stats['raw_recovery'] = dr.groupby('relation').is_recovered.mean()
    rq2_rows.append(t.merge(stats, left_on='relation', right_index=True))

pd.concat(desc_rows).to_csv(OUT / 'arc_descriptives.csv', index=False)
pd.concat(rq1_rows).to_csv(OUT / 'rq1_models.csv', index=False)
pd.DataFrame(slope_rows).to_csv(OUT / 'rq1_slopes.csv', index=False)
pd.DataFrame(vif_rows).to_csv(OUT / 'rq1_collinearity.csv', index=False)
if HAS_ORD:
    pd.concat(rq1o_rows).to_csv(OUT / 'rq1_models_ordered.csv', index=False)
    pd.DataFrame(slopeo_rows).to_csv(OUT / 'rq1_slopes_ordered.csv', index=False)
pd.DataFrame(rq2_tests).to_csv(OUT / 'rq2_tests.csv', index=False)
pd.concat(rq2_rows).sort_values(['dataset', 'OR']).to_csv(OUT / 'rq2_relation_ors.csv', index=False)

# RQ3
s = sents.dropna(subset=['gold_adl_w', 'pred_adl_w', 'copy_adl_w']).copy()
s['gap'] = s.pred_adl_w - s.gold_adl_w
s['copy_gap'] = s.copy_adl_w - s.gold_adl_w
s['disp'] = s.input_adl_bpe - s.gold_adl_bpe
s['excess_copy'] = s.copy_bigram - s.gold_bigram_in_input


def ols(formula, d):
    return smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['item'])[0]})


rq3, rq3_means = [], []
for ds, d in s.groupby('dataset'):
    for p, dp in list(d.groupby('perturbation', observed=True)) + [('pooled', d)]:
        f_extra = ' + C(perturbation)' if p == 'pooled' else ''
        for subset, dd in [('all', dp), ('non_exact', dp[dp.em == 0])]:
            if len(dd) < 30:
                continue
            dd = dd.assign(disp_z=(dd.disp - dd.disp.mean()) / dd.disp.std())
            m = ols('gap ~ disp_z' + f_extra, dd)
            mc = ols('copy_gap ~ disp_z' + f_extra, dd)
            mx = ols('gap ~ disp_z + excess_copy' + f_extra, dd)
            g = dd.groupby('item').gap.mean()
            rq3.append(dict(dataset=ds, perturbation=p, subset=subset, n=len(dd),
                            mean_gap=g.mean(), gap_lo=g.mean() - 1.96 * g.std() / np.sqrt(len(g)),
                            gap_hi=g.mean() + 1.96 * g.std() / np.sqrt(len(g)),
                            b_disp=m.params['disp_z'], p_disp=m.pvalues['disp_z'], r2=m.rsquared,
                            b_copy_baseline=mc.params['disp_z'], r2_copy_baseline=mc.rsquared,
                            b_disp_ctrl_copy=mx.params['disp_z'], p_disp_ctrl_copy=mx.pvalues['disp_z'],
                            b_excess_copy=mx.params['excess_copy'], p_excess_copy=mx.pvalues['excess_copy'],
                            mean_disp=dd.disp.mean(), mean_excess_copy=dd.excess_copy.mean(),
                            mean_pred_adl=dd.pred_adl_w.mean(), mean_gold_adl=dd.gold_adl_w.mean(),
                            em=dd.em.mean()))
pd.DataFrame(rq3).to_csv(OUT / 'rq3_results.csv', index=False)

# Output-length control: is the ADL gap a by-product of shorter outputs?
len_rows = []
for (ds, p), g in s.groupby(['dataset', 'perturbation'], observed=True):
    parts = []
    for seed in g.seed.unique():
        f = ROOT / 'evaluation_results' / ds / p / str(seed) / f'full_samples_1k_{ds}_{p}_final.csv'
        if not f.exists():
            continue
        d = pd.read_csv(f)
        parts.append(pd.DataFrame({'sent_id': range(len(d)), 'seed': seed,
                                   'dlen': d.prediction.astype(str).str.split().str.len()
                                   - d.actual.astype(str).str.split().str.len()}))
    if not parts:
        continue
    g = g.merge(pd.concat(parts), on=['sent_id', 'seed'])
    m = ols('gap ~ dlen', g)
    same = g[g.dlen == 0]
    len_rows.append(dict(dataset=ds, perturbation=p, mean_dlen=g.dlen.mean(), mean_gap=g.gap.mean(),
                         gap_at_dlen0=m.params['Intercept'], lo=m.conf_int().loc['Intercept', 0],
                         hi=m.conf_int().loc['Intercept', 1], gap_same_len=same.gap.mean(), n_same_len=len(same)))
pd.DataFrame(len_rows).to_csv(OUT / 'rq3_length_control.csv', index=False)

arcs[['dataset', 'perturbation', 'seed', 'sent_id', 'item', 'is_recovered', 'log_dist', 'log_gold', 'sent_len_z']].to_csv(OUT / 'arcs_for_glmer.csv', index=False)
print('done')
