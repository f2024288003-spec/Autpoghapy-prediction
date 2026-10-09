"""
08_decision_boundary.py -- UMAP decision landscapes (manuscript Section 3.2.1, Figures 14-16)
--------------------------------------------------------------------------------------------
Same method and settings as Cell 8 of 04_analysis_ROC6_only.ipynb:
    UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=0)
    fitted once on the z-scored training + independent-test features (1,433-d);
    P(class = 1) of every protein is spread over a 200 x 200 grid with a k-nearest-neighbour
    regressor (k = 15), separately for the training and the test panel; black line = 0.5 contour.

Changes:
  * features go through the SAME scaler that was used for training.  The notebook first applied
    sanitize_features(), which clips raw PSCV values to +/-1e6 before scaling; the classifiers never
    saw such inputs during training, so the plotted probabilities were not the models' real outputs.
  * panels are lettered (A, B), point colours have a legend, a colour bar is added, 300 dpi + PDF.
  * the 2-D projection is cached (umap_projection.npy) and reused for every model.

Colab:  !python 08_decision_boundary.py --work_dir /content/drive/MyDrive/pscv_work
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsRegressor
from pscv_common import load_features, load_clf

BLUE, RED = '#3b6fff', '#ff4040'      # 0 = non-autophagy, 1 = autophagy-related (as in the notebook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work_dir', default='/content/drive/MyDrive/pscv_work')
    ap.add_argument('--models', nargs='+', default=['BlockMoE', 'XGBoost', 'MLP'])
    ap.add_argument('--grid', type=int, default=200)
    ap.add_argument('--k', type=int, default=15)
    args = ap.parse_args()

    Xtr, ytr, Xte, yte, _, _ = load_features(args.work_dir)
    out_dir = os.path.join(args.work_dir, 'plots', 'boundary_v2'); os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(args.work_dir, 'umap_projection.npy')
    n_tr = len(Xtr)

    proj_all = np.load(cache) if os.path.exists(cache) else None
    for mname in args.models:
        clf, comb = load_clf(args.work_dir, mname)
        Xtr_s, Xte_s = comb.transform(Xtr), comb.transform(Xte)
        if proj_all is None:      # the scaler is fitted on the same training matrix for every model
            import umap
            proj_all = umap.UMAP(n_components=2, random_state=0, n_neighbors=15, min_dist=0.1,
                                 metric='euclidean').fit_transform(np.vstack([Xtr_s, Xte_s]))
            np.save(cache, proj_all)
        proj_tr, proj_te = proj_all[:n_tr], proj_all[n_tr:]
        p_tr, p_te = clf.predict_proba(Xtr_s), clf.predict_proba(Xte_s)

        x0, x1 = proj_all[:, 0].min() - 1, proj_all[:, 0].max() + 1
        y0, y1 = proj_all[:, 1].min() - 1, proj_all[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x0, x1, args.grid), np.linspace(y0, y1, args.grid))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz_tr = KNeighborsRegressor(n_neighbors=args.k).fit(proj_tr, p_tr).predict(grid).reshape(xx.shape)
        zz_te = KNeighborsRegressor(n_neighbors=args.k).fit(proj_te, p_te).predict(grid).reshape(xx.shape)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True, sharey=True)
        for ax, zz, proj, y, ttl, letter in [
                (axes[0], zz_tr, proj_tr, ytr, f'Training subset (70%, n = {n_tr:,})', 'A'),
                (axes[1], zz_te, proj_te, yte, f'Independent-test subset (30%, n = {len(yte):,})', 'B')]:
            cf = ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 21), cmap=plt.cm.RdBu_r, alpha=0.6)
            ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=1.2)
            ax.scatter(proj[:, 0], proj[:, 1], c=np.where(y == 1, RED, BLUE), s=9, edgecolor='k', linewidth=0.2)
            ax.set_title(ttl, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel('UMAP 1'); ax.text(0.015, 0.975, letter, transform=ax.transAxes, fontsize=18,
                                             fontweight='bold', va='top')
        axes[0].set_ylabel('UMAP 2')
        handles = [Line2D([], [], marker='o', ls='', mfc=RED, mec='k', mew=0.3, label='Autophagy-related (class 1)'),
                   Line2D([], [], marker='o', ls='', mfc=BLUE, mec='k', mew=0.3, label='Non-autophagy (class 0)'),
                   Line2D([], [], color='k', lw=1.2, label='Decision boundary (P = 0.5)')]
        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=11)
        fig.tight_layout(rect=(0, 0.05, 0.93, 1))
        cax = fig.add_axes([0.94, 0.14, 0.012, 0.74])
        fig.colorbar(cf, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0]).set_label('Predicted P(autophagy-related)')
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(out_dir, f'boundary_{mname}.{ext}'), dpi=300)
        plt.close(fig)
        print('wrote', os.path.join(out_dir, f'boundary_{mname}.png'), flush=True)


if __name__ == '__main__':
    main()
