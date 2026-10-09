"""
06_ablation.py -- feature-group ablation on the independent test set (manuscript Section 3.1.4, Table 5)
------------------------------------------------------------------------------------------------------
Retrains each classifier on  (a) PSCV only (153-d)  (b) ESM-2 only (1,280-d)  (c) PSCV + ESM-2 (1,433-d)
with the SAME 70/30 split, the same internal 10% early-stopping split, the same per-block z-scoring
and the same hyper-parameters as 03_classification.ipynb.

Because 03_classification.ipynb does not fix the PyTorch seed, single runs differ slightly from one
another.  To separate a real feature effect from run-to-run noise every neural configuration is trained
with several seeds (default 5) and mean +/- SD is reported.  XGBoost is deterministic (random_state=0).

WideDeep needs two blocks (wide = PSCV, deep = ESM-2) so it has no single-block variant and is skipped.

Outputs (WORK_DIR/ablation/):
    ablation_runs.csv        one row per model x feature set x seed
    ablation_summary.csv     mean +/- SD per model x feature set   (-> Table 5 / Supplementary Table S3)
    ablation_delong.csv      paired DeLong test, fused vs single block (seed-averaged probabilities)
    ablation_preds.pkl       all test-set probabilities

Colab (GPU):  !python 06_ablation.py --work_dir /content/drive/MyDrive/pscv_work
              add  --models BlockMoE XGBoost MLP CNN BayesianMLP   and/or   --seeds 42 43 44 45 46
"""
import argparse, os, pickle, time
import numpy as np, pandas as pd, torch
from pscv_common import (load_features, train_cfg_for, build_classifier, FeatureCombiner,
                         split_internal_val, compute_metrics, seed_everything, RANDOM_SEED)
import importlib.util, sys
_spec = importlib.util.spec_from_file_location('stats05', os.path.join(os.path.dirname(os.path.abspath(__file__)), '05_statistical_tests.py'))
stats05 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(stats05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work_dir', default='/content/drive/MyDrive/pscv_work')
    ap.add_argument('--models', nargs='+', default=['BlockMoE', 'XGBoost', 'MLP', 'CNN', 'BayesianMLP'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument('--sets', nargs='+', default=['PSCV', 'ESM2', 'FUSED'])
    args = ap.parse_args()

    Xtr, ytr, Xte, yte, block_sizes, block_names = load_features(args.work_dir)
    n_pscv = block_sizes[0]
    SETS = {'PSCV':  (slice(0, n_pscv),            [n_pscv],               ['PSCV']),
            'ESM2':  (slice(n_pscv, Xtr.shape[1]), [sum(block_sizes[1:])], ['ESM2']),
            'FUSED': (slice(0, Xtr.shape[1]),      block_sizes,            block_names)}
    cfg = train_cfg_for(ytr)
    tr_idx, val_idx = split_internal_val(ytr, seed=RANDOM_SEED)       # same split as Step 3

    out_dir = os.path.join(args.work_dir, 'ablation'); os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, 'ablation_preds.pkl')
    preds = pickle.load(open(pred_path, 'rb')) if os.path.exists(pred_path) else {}   # resumable

    for mname in args.models:
        for sname in args.sets:
            sl, bs, bn = SETS[sname]
            seeds = [0] if mname == 'XGBoost' else args.seeds
            for seed in seeds:
                key = (mname, sname, seed)
                if key in preds:
                    continue
                comb = FeatureCombiner(bs, bn)
                Xtr_s = comb.fit_transform(Xtr[:, sl]); Xte_s = comb.transform(Xte[:, sl])
                seed_everything(seed)
                t0 = time.time()
                clf = build_classifier(mname, Xtr_s.shape[1], cfg, bs)
                clf.fit(Xtr_s[tr_idx], ytr[tr_idx], Xtr_s[val_idx], ytr[val_idx])
                preds[key] = np.asarray(clf.predict_proba(Xte_s), dtype=np.float64)
                m = compute_metrics(yte, preds[key])
                print(f'{mname:12s} {sname:5s} seed={seed}  AUC={m["AUC"]:.4f} ACC={m["ACC"]:.4f} '
                      f'MCC={m["MCC"]:.4f}  ({time.time()-t0:.0f}s)', flush=True)
                pickle.dump(preds, open(pred_path, 'wb'))

    rows = [dict(model=k[0], features=k[1], seed=k[2], **compute_metrics(yte, p)) for k, p in preds.items()]
    runs = pd.DataFrame(rows); runs.to_csv(os.path.join(out_dir, 'ablation_runs.csv'), index=False)
    metrics = ['AUC', 'ACC', 'MCC', 'F1', 'SENS', 'SPEC']
    g = runs.groupby(['model', 'features'])
    summ = g[metrics].mean().add_suffix('_mean').join(g[metrics].std(ddof=0).add_suffix('_sd')).join(g.size().rename('n_runs')).reset_index()
    summ.to_csv(os.path.join(out_dir, 'ablation_summary.csv'), index=False)

    # paired DeLong on seed-averaged probabilities (one prediction vector per configuration)
    drow = []
    for mname in runs.model.unique():
        avg = {s: np.mean([p for k, p in preds.items() if k[0] == mname and k[1] == s], axis=0)
               for s in SETS if any(k[0] == mname and k[1] == s for k in preds)}
        for s in ('PSCV', 'ESM2'):
            if 'FUSED' in avg and s in avg:
                a1, a2, z, p = stats05.delong_test(yte.astype(int), avg['FUSED'], avg[s])
                drow.append(dict(model=mname, comparison=f'FUSED vs {s}', AUC_fused=a1, AUC_single=a2,
                                 dAUC=a1 - a2, z=z, p=p))
    dl = pd.DataFrame(drow); dl.to_csv(os.path.join(out_dir, 'ablation_delong.csv'), index=False)

    pd.set_option('display.width', 220); pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print('\n', summ.to_string(index=False)); print('\n', dl.to_string(index=False))


if __name__ == '__main__':
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    main()
