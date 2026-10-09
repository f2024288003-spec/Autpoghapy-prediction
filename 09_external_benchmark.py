"""
09_external_benchmark.py -- evaluate the saved classifiers on an EXTERNAL benchmark (optional)
---------------------------------------------------------------------------------------------
Purpose: the head-to-head comparison requested for Section 3.4.  Give it the positive / negative
FASTA files of a published benchmark (e.g. the ATGPred-FL / EnsembleDL-ATG independent test set,
which PLM-ATG also uses) and it will
    1. remove external proteins that are already in OUR training set (identical sequence) -> no leakage
    2. compute PSCV (153-d) + mean-pooled ESM-2 (1,280-d) features exactly as 02_feature_extraction.ipynb
    3. score them with the six saved classifiers (clf_<MODEL>_independent.pkl) -- NO retraining
    4. write AUC / ACC / MCC / F1 / SENS / SPEC (+ DeLong and Wilson 95% CIs) to
       WORK_DIR/external/<name>_metrics.csv

Needs a GPU runtime for ESM-2.  The PSCV part was checked against features.npz; the ESM-2 part is
the same code as notebook 02 but has NOT been run end-to-end by me, so check the first lines of output.

Colab:
  !python 09_external_benchmark.py --work_dir /content/drive/MyDrive/pscv_work \
          --pos /content/ext_pos.fasta --neg /content/ext_neg.fasta --name ATGPredFL_test
"""
import argparse, os
import numpy as np, pandas as pd, torch
from pscv_common import load_clf, compute_metrics, PAPER_MODELS, DEVICE
import importlib.util
_spec = importlib.util.spec_from_file_location('stats05', os.path.join(os.path.dirname(os.path.abspath(__file__)), '05_statistical_tests.py'))
stats05 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(stats05)

# ======================= PSCV (verbatim from 02_feature_extraction.ipynb, Cell 4) =======================
import math
import numpy as np

AA_ORDER = ['A','C','D','E','F','G','H','I','K','L',
            'M','N','P','Q','R','S','T','V','W','Y','X']
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}
N_AA = len(AA_ORDER)
ALLOWED = set(AA_ORDER)


def seq_to_mat(seq):
    mat = np.zeros((N_AA, N_AA), dtype=np.float64)
    for k in range(len(seq) - 1):
        a, b = seq[k], seq[k+1]
        if a in AA_INDEX and b in AA_INDEX:
            mat[AA_INDEX[a], AA_INDEX[b]] += 1
    return mat


def prim(seq):
    mat = np.zeros((N_AA, N_AA), dtype=np.float64)
    for i, aa1 in enumerate(AA_ORDER):
        first = -1
        for x, ch in enumerate(seq):
            if ch == aa1:
                first = x + 1
                break
        if first == -1:
            continue
        for j, aa2 in enumerate(AA_ORDER):
            if i == j: continue
            s = 0
            for y, ch in enumerate(seq):
                if ch == aa2:
                    s += (y + 1) - first
            mat[i, j] = s
    return mat


def frequency_vec(seq):
    fv = [0] * N_AA
    for ch in seq:
        if ch in AA_INDEX:
            fv[AA_INDEX[ch]] += 1
    return fv


def aapiv(seq):
    encoder = ['X'] + AA_ORDER[:-1]
    apv = [0] * N_AA
    for i in range(N_AA):
        s = 0
        for j, ch in enumerate(seq):
            if ch == encoder[i]:
                s += j + 1
        apv[i] = s
    return apv[1:] + apv[:1]


def raw_moments(mat, order=3):
    n = mat.shape[0]
    out = []
    idx = np.arange(1, n+1, dtype=np.float64)
    for i in range(order+1):
        for j in range(order+1):
            if i + j <= order:
                out.append(float((idx**i) @ mat @ (idx**j)))
    return out


def central_moments(mat, order, xbar, ybar):
    n = mat.shape[0]
    out = []
    idx = np.arange(1, n+1, dtype=np.float64)
    for i in range(order+1):
        for j in range(order+1):
            if i + j <= order:
                out.append(float(((idx - xbar)**i) @ mat @ ((idx - ybar)**j)))
    return out


def _poch(a, k):
    ans = 1.0
    for i in range(k):
        ans *= (a + i)
    return ans


def _gamma(x):
    return math.exp(math.lgamma(x))


def _hahn_processor(x, n, N):
    rho = _gamma(n+1.0) * _gamma(n+1.0) * _poch(n+1.0, N)
    if rho < 0: rho = 0.0
    ans1 = _poch(N-1.0, n) * _poch(N-1.0, n)
    ans2 = 0.0
    for k in range(n+1):
        ans2 += ((-1.0)**k) * (_poch(-n, k) * _poch(-x, k) * _poch(2*N - n - 1.0, k))
    return (ans1 + ans2) * math.sqrt(rho)


def _hahn_moment(m, n, N, mat):
    val = 0.0
    for x in range(N):
        for y in range(N):
            val += mat[x, y] * _hahn_processor(x, m, N) * _hahn_processor(x, n, N)
    return val


def hahn_moments(mat, order=3):
    N = mat.shape[0]
    out = []
    for i in range(order+1):
        for j in range(order+1):
            if i + j <= order:
                out.append(_hahn_moment(i, j, N, mat))
    return out


def pscv_features(seq, order=3):
    seq = seq.upper()
    seq = ''.join(ch if ch in ALLOWED else 'X' for ch in seq)
    fv = []
    M = seq_to_mat(seq)
    rm = raw_moments(M, order); fv.extend(rm)
    fv.extend(central_moments(M, order, rm[4], rm[1]))
    fv.extend(hahn_moments(M, order))
    fv.extend(frequency_vec(seq))
    P = prim(seq)
    rmP = raw_moments(P, order); fv.extend(rmP)
    fv.extend(central_moments(P, order, rmP[4], rmP[1]))
    fv.extend(hahn_moments(P, order))
    fv.extend(aapiv(seq))
    R = prim(seq[::-1])
    rmR = raw_moments(R, order); fv.extend(rmR)
    fv.extend(central_moments(R, order, rmR[4], rmR[1]))
    fv.extend(hahn_moments(R, order))
    fv.extend(aapiv(seq[::-1]))
    arr = np.asarray(fv, dtype=np.float64)
    assert arr.shape[0] == 153, f'expected 153, got {arr.shape[0]}'
    return arr


def pscv_group_indices():
    return {
        'seqMat_raw':     slice(0,   10),
        'seqMat_central': slice(10,  20),
        'seqMat_hahn':    slice(20,  30),
        'freq_vec':       slice(30,  51),
        'PRIM_raw':       slice(51,  61),
        'PRIM_central':   slice(61,  71),
        'PRIM_hahn':      slice(71,  81),
        'AAPIV':          slice(81,  102),
        'RPRIM_raw':      slice(102, 112),
        'RPRIM_central':  slice(112, 122),
        'RPRIM_hahn':     slice(122, 132),
        'RAAPIV':         slice(132, 153),
    }


# ======================= ESM-2 (same as 02_feature_extraction.ipynb, Cell 7) =======================
PLM_MODEL, PLM_DIM, MAX_LEN, BATCH = 'facebook/esm2_t33_650M_UR50D', 1280, 1022, 4


@torch.no_grad()
def embed_sequences(seqs):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(PLM_MODEL, do_lower_case=False)
    plm = AutoModel.from_pretrained(PLM_MODEL).to(DEVICE).eval()
    out = np.zeros((len(seqs), PLM_DIM), dtype=np.float32)
    for n, s in enumerate(seqs):
        s = ''.join(('X' if c in 'UZOB' else c) for c in s.upper())
        wins = [s] if len(s) <= MAX_LEN else [s[i:i + MAX_LEN] for i in range(0, len(s), MAX_LEN)]
        means = []
        for i in range(0, len(wins), BATCH):
            enc = tok(wins[i:i + BATCH], return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN + 2)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            hs = plm(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).float()
            means.append(((hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)).cpu().numpy())
        out[n] = np.concatenate(means, 0).mean(0)
        if n % 50 == 0: print(f'  ESM-2 {n}/{len(seqs)}', flush=True)
    return out


def read_fasta(path, label):
    from Bio import SeqIO
    std = set('ACDEFGHIKLMNPQRSTVWYX')
    rows = []
    for rec in SeqIO.parse(path, 'fasta'):
        s = ''.join(('X' if c in 'UZOB' else c) for c in str(rec.seq).upper().strip())
        if s and all(c in std for c in s):
            rows.append(dict(id=rec.id, sequence=s, label=label))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work_dir', default='/content/drive/MyDrive/pscv_work')
    ap.add_argument('--pos', required=True); ap.add_argument('--neg', required=True)
    ap.add_argument('--name', default='external')
    ap.add_argument('--keep_overlap', action='store_true', help='do not drop proteins present in our training set')
    args = ap.parse_args()

    df = pd.DataFrame(read_fasta(args.pos, 1) + read_fasta(args.neg, 0)).drop_duplicates('sequence')
    n0 = len(df)
    if not args.keep_overlap:
        train_seqs = set(pd.read_csv(os.path.join(args.work_dir, 'train.csv'))['sequence'])
        df = df[~df['sequence'].isin(train_seqs)].reset_index(drop=True)
    print(f'{args.name}: {n0} sequences read, {len(df)} kept after removing training-set duplicates '
          f'(pos={int(df.label.sum())}, neg={int((df.label == 0).sum())})')

    out_dir = os.path.join(args.work_dir, 'external'); os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, f'{args.name}_features.npz')
    if os.path.exists(cache):
        X = np.load(cache)['X']
    else:
        Xp = np.stack([pscv_features(s).astype(np.float32) for s in df['sequence']])
        X = np.concatenate([Xp, embed_sequences(df['sequence'].tolist())], axis=1).astype(np.float32)
        np.savez_compressed(cache, X=X, y=df['label'].values)
    y = df['label'].values.astype(int)

    rows = []
    for m in PAPER_MODELS:
        clf, comb = load_clf(args.work_dir, m)
        p = np.asarray(clf.predict_proba(comb.transform(X)), dtype=np.float64)
        met = compute_metrics(y, p)
        _, _, lo, hi = stats05.delong_ci(y, p)
        alo, ahi = stats05.wilson_ci(int(round(met['ACC'] * len(y))), len(y))
        rows.append(dict(model=m, n=len(y), **met, AUC_lo=lo, AUC_hi=hi, ACC_lo=alo, ACC_hi=ahi))
        df[f'prob_{m}'] = p
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, f'{args.name}_metrics.csv'), index=False)
    df.drop(columns='sequence').to_csv(os.path.join(out_dir, f'{args.name}_predictions.csv'), index=False)
    pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(res.to_string(index=False))


if __name__ == '__main__':
    main()
