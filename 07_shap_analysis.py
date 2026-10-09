"""
07_shap_analysis.py -- SHAP for BlockMoE, XGBoost and MLP (manuscript Section 3.2.2, Figures 17-19)
--------------------------------------------------------------------------------------------------
Differences from Cell 10 of 04_analysis_ROC6_only.ipynb (everything else is the same):
  * neural models are explained in eval() mode (dropout off, BatchNorm running statistics);
    the notebook left them in train() mode, which makes the attributions depend on the batch
  * more samples: 200 background (training) and 1,000 explained (independent-test) proteins
  * SHAP is summarised over ALL 1,433 features: block level (PSCV vs ESM-2), PSCV group level,
    and the top-30 PSCV features (the figure used in the paper)
  * BlockMoE gate weights (how much the PSCV / ESM-2 expert contributes) are reported

Outputs (WORK_DIR/plots/shap_v2/ and WORK_DIR/shap_v2/):
    shap_pscv_<MODEL>.png|.pdf     top-30 PSCV features            (Figures 17-19)
    shap_blocks_all.png|.pdf       share of total |SHAP| per block (new, optional figure)
    shap_summary.json              every number quoted in the text
    shap_values_<MODEL>.npy        raw SHAP matrix (n_explained x 1433)

Colab:  !python 07_shap_analysis.py --work_dir /content/drive/MyDrive/pscv_work
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn, shap
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pscv_common import (load_features, load_clf, PSCV_GROUPS, AA_ORDER, pscv_feature_names,
                         XGBoostClassifier, DEVICE)

GROUP_OF = {'seqMat': 'SeqMat moments', 'freq_vec': 'FreqVec', 'PRIM': 'PRIM moments', 'AAPIV': 'AAPIV',
            'RPRIM': 'RPRIM moments', 'RAAPIV': 'RAAPIV'}


def pretty(name):
    g, k = name.split('[')[0], int(name.split('[')[1][:-1])
    if g in ('freq_vec', 'AAPIV', 'RAAPIV'):
        return f'{g}[{k}] ({AA_ORDER[k]})'
    return name


def compute_shap(clf, bg, ex):
    if isinstance(clf, XGBoostClassifier):
        sv = shap.TreeExplainer(clf.model).shap_values(ex)          # log-odds scale
        return np.asarray(sv[-1] if isinstance(sv, list) else sv), 'TreeExplainer (log-odds)'

    class ProbWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return torch.sigmoid(self.m(x)).unsqueeze(-1)

    wrapped = ProbWrap(clf.torch_module()).to(DEVICE).eval()        # eval(): deterministic forward pass
    expl = shap.GradientExplainer(wrapped, torch.from_numpy(bg).float().to(DEVICE))
    vals = expl.shap_values(torch.from_numpy(ex).float().to(DEVICE), nsamples=200, rseed=0)
    if isinstance(vals, list): vals = vals[0]
    vals = np.asarray(vals)
    if vals.ndim == 3: vals = vals[..., 0]
    return vals, 'GradientExplainer (probability)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work_dir', default='/content/drive/MyDrive/pscv_work')
    ap.add_argument('--models', nargs='+', default=['XGBoost', 'MLP', 'BlockMoE'])
    ap.add_argument('--n_background', type=int, default=200)
    ap.add_argument('--n_explain', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    Xtr, ytr, Xte, yte, block_sizes, block_names = load_features(args.work_dir)
    n_pscv = block_sizes[0]
    names = pscv_feature_names()
    plot_dir = os.path.join(args.work_dir, 'plots', 'shap_v2'); os.makedirs(plot_dir, exist_ok=True)
    data_dir = os.path.join(args.work_dir, 'shap_v2'); os.makedirs(data_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    bg_idx = rng.choice(len(Xtr), size=min(args.n_background, len(Xtr)), replace=False)
    ex_idx = rng.choice(len(Xte), size=min(args.n_explain, len(Xte)), replace=False)
    summary = {'n_background': int(len(bg_idx)), 'n_explained': int(len(ex_idx)), 'seed': args.seed, 'models': {}}

    for mname in args.models:
        clf, comb = load_clf(args.work_dir, mname)
        Xtr_s, Xte_s = comb.transform(Xtr), comb.transform(Xte)
        sv, method = compute_shap(clf, Xtr_s[bg_idx], Xte_s[ex_idx])
        np.save(os.path.join(data_dir, f'shap_values_{mname}.npy'), sv.astype(np.float32))
        mean_abs = np.abs(sv).mean(axis=0)                       # (1433,)
        total = float(mean_abs.sum())
        pscv_share = float(mean_abs[:n_pscv].sum() / total)

        groups = {}
        for g, sl in PSCV_GROUPS.items():
            key = GROUP_OF[g.split('_')[0] if g != 'freq_vec' else 'freq_vec']
            groups[key] = groups.get(key, 0.0) + float(mean_abs[sl].sum())
        pscv_total = sum(groups.values())
        order_all = np.argsort(mean_abs)[::-1]
        top30 = np.argsort(mean_abs[:n_pscv])[::-1][:30]
        cum = np.cumsum(np.sort(mean_abs[:n_pscv])[::-1]) / pscv_total

        info = {
            'explainer': method,
            'share_of_total_abs_shap': {'PSCV (153)': pscv_share, 'ESM-2 (1280)': 1 - pscv_share},
            'mean_abs_shap_per_feature': {'PSCV': float(mean_abs[:n_pscv].mean()), 'ESM-2': float(mean_abs[n_pscv:].mean())},
            'n_pscv_among_top30_of_all_features': int((order_all[:30] < n_pscv).sum()),
            'n_pscv_among_top100_of_all_features': int((order_all[:100] < n_pscv).sum()),
            'best_rank_of_a_pscv_feature': int(np.where(order_all < n_pscv)[0][0]) + 1,
            'pscv_group_share_within_pscv': {k: v / pscv_total for k, v in groups.items()},
            'n_pscv_features_for_50pct_of_pscv_attribution': int(np.searchsorted(cum, 0.5) + 1),
            'n_pscv_features_for_80pct_of_pscv_attribution': int(np.searchsorted(cum, 0.8) + 1),
            'top30_pscv': [(pretty(names[i]), float(mean_abs[i])) for i in top30],
            'top30_groups_count': {},
        }
        for i in top30:
            g = names[i].split('[')[0]; key = GROUP_OF[g.split('_')[0] if g != 'freq_vec' else 'freq_vec']
            info['top30_groups_count'][key] = info['top30_groups_count'].get(key, 0) + 1

        if mname == 'BlockMoE':                                   # how much does each expert contribute?
            mod = clf.torch_module().eval()
            with torch.no_grad():
                xt = torch.from_numpy(Xte_s).float().to(DEVICE)
                w = mod.gate_weights(xt).cpu().numpy()
                e = torch.stack([ex(xt[:, sl]) for ex, sl in zip(mod.experts, mod.slices)], 1).cpu().numpy()
            info['gate_weight_mean'] = {'PSCV expert': float(w[:, 0].mean()), 'ESM-2 expert': float(w[:, 1].mean())}
            info['gate_weight_sd'] = {'PSCV expert': float(w[:, 0].std()), 'ESM-2 expert': float(w[:, 1].std())}
            info['mean_abs_weighted_logit'] = {'PSCV expert': float(np.abs(w[:, 0] * e[:, 0]).mean()),
                                               'ESM-2 expert': float(np.abs(w[:, 1] * e[:, 1]).mean())}
        summary['models'][mname] = info

        fig, ax = plt.subplots(figsize=(7.5, 8.5))
        ax.barh([pretty(names[i]) for i in top30][::-1], mean_abs[top30][::-1], color='#d9730d')
        ax.set_xlabel('mean(|SHAP value|)' + (' (log-odds)' if mname == 'XGBoost' else ' (probability)'))
        ax.set_title(f'{mname}: 30 most influential PSCV features')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        ax.grid(axis='x', alpha=0.3); fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(plot_dir, f'shap_pscv_{mname}.{ext}'), dpi=300)
        plt.close(fig)
        print(f'{mname}: PSCV share of total |SHAP| = {100*pscv_share:.1f}%  '
              f'(best PSCV rank among 1433 features: {info["best_rank_of_a_pscv_feature"]})', flush=True)

    ms = list(summary['models'])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    p = [100 * summary['models'][m]['share_of_total_abs_shap']['PSCV (153)'] for m in ms]
    ax.barh(ms, p, color='#d9730d', label='PSCV (153 features)')
    ax.barh(ms, [100 - v for v in p], left=p, color='#3b6fb6', label='ESM-2 (1,280 features)')
    for i, v in enumerate(p):
        ax.text(v + 1.5, i, f'PSCV {v:.1f}%', va='center', color='white', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 100); ax.set_xlabel('Share of total mean(|SHAP|) (%)'); ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False); fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(plot_dir, f'shap_blocks_all.{ext}'), dpi=300)
    plt.close(fig)

    json.dump(summary, open(os.path.join(data_dir, 'shap_summary.json'), 'w'), indent=2)
    print('saved ->', plot_dir, 'and', data_dir)


if __name__ == '__main__':
    main()
