"""
05_statistical_tests.py  --  DeLong + McNemar tests on the saved independent-test predictions
-------------------------------------------------------------------------------------------
Reads   WORK_DIR/results.pkl   (written by 03_classification.ipynb)
Writes  WORK_DIR/stats/auc_ci_delong.csv        AUC with DeLong 95% CI, ACC with Wilson 95% CI
        WORK_DIR/stats/pairwise_delong.csv      pairwise DeLong z, p (raw + Holm-adjusted)
        WORK_DIR/stats/pairwise_mcnemar.csv     pairwise McNemar b, c, p (raw + Holm-adjusted)

No retraining is needed: every model was scored on the SAME 4,684 test proteins,
so the tests are paired.

Colab:   !python 05_statistical_tests.py --work_dir /content/drive/MyDrive/pscv_work
"""
import argparse, itertools, os, pickle
import numpy as np
import pandas as pd
from scipy import stats

MODELS = ['BlockMoE', 'XGBoost', 'WideDeep', 'MLP', 'CNN', 'BayesianMLP']   # the six in the paper


# ---------------- DeLong (fast algorithm, Sun & Xu 2014) ----------------
def _midrank(x):
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_components(y_true, probs):
    """probs: (k_models, n). Returns AUC vector (k,) and covariance matrix (k, k)."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.atleast_2d(probs)
    pos, neg = probs[:, y_true == 1], probs[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = probs.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_test(y_true, p1, p2):
    aucs, cov = delong_components(y_true, np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def delong_ci(y_true, p, alpha=0.05):
    aucs, cov = delong_components(y_true, p[None, :])
    se = float(np.sqrt(cov[0, 0]))
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(aucs[0]), se, float(aucs[0] - zc * se), float(min(1.0, aucs[0] + zc * se))


# ---------------- McNemar ----------------
def mcnemar_test(y_true, p1, p2, thr=0.5):
    y = np.asarray(y_true).astype(int)
    c1 = ((p1 >= thr).astype(int) == y)
    c2 = ((p2 >= thr).astype(int) == y)
    b = int(np.sum(c1 & ~c2))      # model 1 right, model 2 wrong
    c = int(np.sum(~c1 & c2))      # model 1 wrong, model 2 right
    nd = b + c
    p_exact = 1.0 if nd == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), nd, 0.5))
    chi2 = 0.0 if nd == 0 else (abs(b - c) - 1) ** 2 / nd     # continuity-corrected
    return b, c, chi2, float(stats.chi2.sf(chi2, 1)), float(p_exact)


def wilson_ci(k, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work_dir', default='/content/drive/MyDrive/pscv_work')
    ap.add_argument('--protocol', default='independent')
    args = ap.parse_args()

    with open(os.path.join(args.work_dir, 'results.pkl'), 'rb') as f:
        res = pickle.load(f)
    out_dir = os.path.join(args.work_dir, 'stats')
    os.makedirs(out_dir, exist_ok=True)

    models = [m for m in MODELS if args.protocol in res.get(m, {})]
    y = np.asarray(res[models[0]][args.protocol]['y_true']).astype(int)
    P = {m: np.asarray(res[m][args.protocol]['y_prob'], dtype=np.float64) for m in models}
    for m in models:
        assert np.array_equal(np.asarray(res[m][args.protocol]['y_true']).astype(int), y), \
            f'{m}: test labels differ -> predictions are not paired'

    rows = []
    for m in models:
        auc, se, lo, hi = delong_ci(y, P[m])
        k = int(((P[m] >= 0.5).astype(int) == y).sum())
        alo, ahi = wilson_ci(k, len(y))
        rows.append(dict(model=m, n=len(y), AUC=auc, AUC_SE=se, AUC_lo=lo, AUC_hi=hi,
                         ACC=k / len(y), ACC_lo=alo, ACC_hi=ahi))
    ci = pd.DataFrame(rows).sort_values('AUC', ascending=False)
    ci.to_csv(os.path.join(out_dir, 'auc_ci_delong.csv'), index=False)

    d_rows, m_rows = [], []
    for a, b in itertools.combinations(models, 2):
        a1, a2, z, p = delong_test(y, P[a], P[b])
        d_rows.append(dict(model_A=a, model_B=b, AUC_A=a1, AUC_B=a2, dAUC=a1 - a2, z=z, p=p))
        bb, cc, chi2, p_chi, p_ex = mcnemar_test(y, P[a], P[b])
        m_rows.append(dict(model_A=a, model_B=b, A_right_B_wrong=bb, A_wrong_B_right=cc,
                           chi2_cc=chi2, p_chi2=p_chi, p_exact=p_ex))
    dl = pd.DataFrame(d_rows); dl['p_holm'] = holm(dl['p'])
    mc = pd.DataFrame(m_rows); mc['p_holm'] = holm(mc['p_exact'])
    dl.to_csv(os.path.join(out_dir, 'pairwise_delong.csv'), index=False)
    mc.to_csv(os.path.join(out_dir, 'pairwise_mcnemar.csv'), index=False)

    pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print('\nAUC (DeLong 95% CI) and ACC (Wilson 95% CI)\n', ci.to_string(index=False))
    print('\nPairwise DeLong test\n', dl.to_string(index=False))
    print('\nPairwise McNemar test\n', mc.to_string(index=False))
    print('\nSaved to', out_dir)


if __name__ == '__main__':
    main()
