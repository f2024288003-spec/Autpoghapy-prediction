"""
pscv_common.py -- shared definitions for the revision scripts (05-08).

Contains exact copies of the classes/functions used in 03_classification.ipynb
(FeatureCombiner, TrainConfig, the six classifiers reported in the paper, the
training loop, metrics) plus a loader that can open the clf_<MODEL>_independent.pkl
checkpoints on CPU or GPU.  Nothing here changes the published pipeline.
"""
import io, os, pickle, random
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, matthews_corrcoef, f1_score,
                             roc_auc_score, confusion_matrix)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PAPER_MODELS = ['BlockMoE', 'XGBoost', 'WideDeep', 'MLP', 'CNN', 'BayesianMLP']
RANDOM_SEED = 42

PSCV_GROUPS = {
    'seqMat_raw': slice(0, 10), 'seqMat_central': slice(10, 20), 'seqMat_hahn': slice(20, 30),
    'freq_vec': slice(30, 51),
    'PRIM_raw': slice(51, 61), 'PRIM_central': slice(61, 71), 'PRIM_hahn': slice(71, 81),
    'AAPIV': slice(81, 102),
    'RPRIM_raw': slice(102, 112), 'RPRIM_central': slice(112, 122), 'RPRIM_hahn': slice(122, 132),
    'RAAPIV': slice(132, 153),
}
AA_ORDER = list('ACDEFGHIKLMNPQRSTVWY') + ['X']   # index k of freq_vec / AAPIV / RAAPIV


def pscv_feature_names():
    names = [''] * 153
    for g, sl in PSCV_GROUPS.items():
        for k, idx in enumerate(range(sl.start, sl.stop)):
            names[idx] = f'{g}[{k}]'
    return names


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------- scaling
def _clean_block(X_block_raw):
    fi = np.finfo(np.float32)
    Xc = np.clip(X_block_raw, fi.min, fi.max).astype(np.float32)
    return np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)


class FeatureCombiner:
    """Per-block StandardScaler (identical to 03_classification.ipynb, Cell 4)."""
    def __init__(self, block_sizes, block_names):
        self.block_sizes = list(block_sizes); self.block_names = list(block_names)
        self.scalers = [StandardScaler() for _ in self.block_sizes]; self.fitted = False

    @property
    def block_slices(self):
        s, start = [], 0
        for b in self.block_sizes:
            s.append(slice(start, start + b)); start += b
        return s

    def fit(self, X):
        for sl, sc in zip(self.block_slices, self.scalers):
            sc.fit(_clean_block(X[:, sl]))
        self.fitted = True
        return self

    def transform(self, X):
        out = np.empty_like(X, dtype=np.float32)
        for sl, sc in zip(self.block_slices, self.scalers):
            out[:, sl] = np.nan_to_num(sc.transform(_clean_block(X[:, sl])),
                                       nan=0.0, posinf=0.0, neginf=0.0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    device: str = DEVICE
    pos_weight: float = 1.0


# ----------------------------------------------------------------- networks
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)


class CNN1DClassifier(nn.Module):
    def __init__(self, in_dim, channels=(64, 128, 64), kernel=5, dropout=0.3):
        super().__init__()
        layers, c_prev = [], 1
        for c in channels:
            layers += [nn.Conv1d(c_prev, c, kernel_size=kernel, padding=kernel // 2),
                       nn.ReLU(), nn.BatchNorm1d(c), nn.MaxPool1d(2), nn.Dropout(dropout)]
            c_prev = c
        self.conv = nn.Sequential(*layers)
        out_len = in_dim
        for _ in channels: out_len //= 2
        self.head = nn.Sequential(nn.Linear(c_prev * out_len, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))
    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)).flatten(1)).squeeze(-1)


class _Expert(nn.Module):
    def __init__(self, d_in, d_hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.BatchNorm1d(d_hidden), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class BlockMoE(nn.Module):
    def __init__(self, block_sizes, d_hidden=256, dropout=0.2):
        super().__init__()
        self.block_sizes = block_sizes; self.slices = []
        s = 0
        for bs in block_sizes:
            self.slices.append(slice(s, s + bs)); s += bs
        self.experts = nn.ModuleList([_Expert(bs, d_hidden, dropout) for bs in block_sizes])
        self.gate = nn.Sequential(nn.Linear(2 * len(block_sizes), 32), nn.GELU(),
                                  nn.Linear(32, len(block_sizes)))
    def gate_weights(self, x):
        stats = []
        for sl in self.slices:
            xb = x[:, sl]
            stats += [xb.mean(dim=1, keepdim=True), xb.std(dim=1, keepdim=True)]
        return torch.softmax(self.gate(torch.cat(stats, dim=1)), dim=1)
    def forward(self, x):
        w = self.gate_weights(x)
        e = torch.stack([ex(x[:, sl]) for ex, sl in zip(self.experts, self.slices)], dim=1)
        return (w * e).sum(dim=1)


class WideDeep(nn.Module):
    def __init__(self, block_sizes, deep_hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        self.wide_dim = block_sizes[0]; self.deep_dim = sum(block_sizes[1:])
        self.wide_slice = slice(0, self.wide_dim)
        self.deep_slice = slice(self.wide_dim, self.wide_dim + self.deep_dim)
        self.wide = nn.Linear(self.wide_dim, 1)
        layers, prev = [], self.deep_dim
        for h in deep_hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)
    def forward(self, x):
        return self.wide(x[:, self.wide_slice]).squeeze(-1) + self.deep(x[:, self.deep_slice]).squeeze(-1)


class BayesianMLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.25):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)


MODEL_REGISTRY = {'MLP': MLPClassifier, 'CNN': CNN1DClassifier}


def build_nn_model(name, in_dim, **kw):
    return MODEL_REGISTRY[name](in_dim=in_dim, **kw)


# ----------------------------------------------------------------- wrappers
class Classifier(ABC):
    name = 'base'
    @abstractmethod
    def fit(self, X_tr, y_tr, X_val, y_val): ...
    @abstractmethod
    def predict_proba(self, X): ...
    def torch_module(self): return None


class NNClassifier(Classifier):
    """Early stopping on validation AUC (identical to 03_classification.ipynb, Cell 7)."""
    def __init__(self, model_name, in_dim, train_cfg, model_kwargs=None):
        self.name = model_name; self.in_dim = in_dim; self.cfg = train_cfg
        self.model_kwargs = model_kwargs or {}; self.model = None
    def _factory(self): return build_nn_model(self.name, self.in_dim, **self.model_kwargs)
    def fit(self, X_tr, y_tr, X_val, y_val):
        model = self._factory().to(self.cfg.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.cfg.pos_weight], device=self.cfg.device))
        ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        best_auc, best_state, stale = -1.0, None, 0
        X_val_t = torch.from_numpy(X_val).float().to(self.cfg.device)
        for ep in range(self.cfg.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.cfg.device), yb.to(self.cfg.device)
                opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(X_val_t)).cpu().numpy()
            try: auc = roc_auc_score(y_val, pv)
            except Exception: auc = 0.0
            if auc > best_auc:
                best_auc, stale = auc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.cfg.early_stopping_patience: break
        if best_state is not None: model.load_state_dict(best_state)
        self.model = model
        return self
    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval(); out = []
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).float().to(self.cfg.device)
            out.append(torch.sigmoid(self.model(xb)).cpu().numpy())
        return np.concatenate(out)
    def torch_module(self): return self.model


class XGBoostClassifier(Classifier):
    name = 'XGBoost'
    def __init__(self, in_dim, train_cfg):
        self.in_dim = in_dim; self.cfg = train_cfg; self.model = None
    def fit(self, X_tr, y_tr, X_val, y_val):
        import xgboost as xgb
        pos = y_tr.sum(); neg = len(y_tr) - pos
        self.model = xgb.XGBClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.7, reg_lambda=1.0, min_child_weight=2,
            scale_pos_weight=float(neg / max(pos, 1)), objective='binary:logistic',
            eval_metric='auc', tree_method='hist', early_stopping_rounds=30,
            n_jobs=-1, random_state=0)
        self.model.fit(X_tr, y_tr.astype(int), eval_set=[(X_val, y_val.astype(int))], verbose=False)
        return self
    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]


class BlockMoEClassifier(NNClassifier):
    name = 'BlockMoE'
    def __init__(self, in_dim, train_cfg, block_sizes):
        self.name = 'BlockMoE'; self.in_dim = in_dim; self.cfg = train_cfg
        self.block_sizes = block_sizes; self.model_kwargs = {}; self.model = None
    def _factory(self): return BlockMoE(self.block_sizes)


class WideDeepClassifier(NNClassifier):
    name = 'WideDeep'
    def __init__(self, in_dim, train_cfg, block_sizes):
        self.name = 'WideDeep'; self.in_dim = in_dim; self.cfg = train_cfg
        self.block_sizes = block_sizes; self.model_kwargs = {}; self.model = None
    def _factory(self): return WideDeep(self.block_sizes)


class BayesianMLPClassifier(NNClassifier):
    name = 'BayesianMLP'
    def __init__(self, in_dim, train_cfg, T=30):
        self.name = 'BayesianMLP'; self.in_dim = in_dim; self.cfg = train_cfg; self.T = T
        self.model_kwargs = {}; self.model = None; self._last_std = None
    def _factory(self): return BayesianMLP(self.in_dim)
    @torch.no_grad()
    def predict_proba(self, X):
        for m in self.model.modules():
            m.train() if isinstance(m, nn.Dropout) else m.eval()
        probs = []
        for _ in range(self.T):
            out = []
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(X[i:i + 256]).float().to(self.cfg.device)
                out.append(torch.sigmoid(self.model(xb)).cpu().numpy())
            probs.append(np.concatenate(out))
        probs = np.stack(probs, 0)
        self._last_std = probs.std(axis=0)
        return probs.mean(axis=0)


def build_classifier(name, in_dim, train_cfg, block_sizes):
    if name in ('MLP', 'CNN'):  return NNClassifier(name, in_dim, train_cfg)
    if name == 'XGBoost':       return XGBoostClassifier(in_dim, train_cfg)
    if name == 'BlockMoE':      return BlockMoEClassifier(in_dim, train_cfg, list(block_sizes))
    if name == 'WideDeep':      return WideDeepClassifier(in_dim, train_cfg, list(block_sizes))
    if name == 'BayesianMLP':   return BayesianMLPClassifier(in_dim, train_cfg)
    raise ValueError(name)


# ----------------------------------------------------------------- protocol helpers
def split_internal_val(y, val_frac=0.1, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y)); rng.shuffle(idx)
    n_val = max(2, int(len(idx) * val_frac))
    pos = idx[y[idx] == 1]; neg = idx[y[idx] == 0]
    n_val_pos = max(1, int(n_val * y.mean())); n_val_neg = max(1, n_val - n_val_pos)
    val_idx = np.concatenate([pos[:n_val_pos], neg[:n_val_neg]])
    return np.setdiff1d(idx, val_idx), val_idx


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int); y_true = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'AUC': roc_auc_score(y_true, y_prob), 'ACC': accuracy_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred), 'F1': f1_score(y_true, y_pred, zero_division=0),
            'SENS': tp / (tp + fn) if (tp + fn) else 0.0, 'SPEC': tn / (tn + fp) if (tn + fp) else 0.0}


def load_features(work_dir):
    d = np.load(os.path.join(work_dir, 'features.npz'), allow_pickle=True)
    return (d['X_train'], d['y_train'].astype(np.float32), d['X_indep'], d['y_indep'].astype(np.float32),
            d['block_sizes'].tolist(), [str(n) for n in d['block_names']])


def train_cfg_for(y_train):
    r = float(y_train.mean())
    return TrainConfig(pos_weight=(1 - r) / max(r, 1e-3))


# ----------------------------------------------------------------- checkpoint loader
class _Unpickler(pickle.Unpickler):
    """Opens notebook-pickled classifiers from a script and on machines without a GPU."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=DEVICE, weights_only=False)
        if module == '__main__' and name in globals():
            return globals()[name]
        return super().find_class(module, name)


def load_clf(work_dir, mname):
    with open(os.path.join(work_dir, f'clf_{mname}_independent.pkl'), 'rb') as f:
        d = _Unpickler(f).load()
    clf, comb = d['classifier'], d['combiner']
    if hasattr(clf, 'cfg'): clf.cfg.device = DEVICE
    if clf.torch_module() is not None: clf.model.to(DEVICE)
    return clf, comb
