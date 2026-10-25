"""Extended / nearest-neighbour baselines (WWW2027 WP4).

Provenance per method (pinned local clones under www2027/external/):
  PTS      faithful PyTorch port of the OFFICIAL TF code
           (tochris/pts-uncertainty @ a4ba144, pts_calibrator.py): input =
           top-10 sorted logits, MLP 2x5 ReLU, T = |out|, loss = MSE between
           calibrated softmax and one-hot labels, Adam lr 5e-5, batch
           min(1000, n), 100k steps (paper recipe), logit clip +-100.
           Native objective; "PTS-BCE" swaps only the loss.
  ETS      Mix-n-Match ensemble temperature scaling via the official
           Zhang et al. util code path as bundled by ThyrixYang/
           group_calibration @ 1fb9c6b (adapted from zhang64-llnl/
           Mix-n-Match-Calibration): T by NLL, 3 weights on
           {softmax(z/T), softmax(z), uniform} by NLL, SLSQP simplex.
  DirODIR  Dirichlet calibration with ODIR regularization (Kull et al.
           2019): W log softmax + b, L-BFGS on NLL + lambda*offdiag(W)^2 +
           mu*b^2; (lambda,mu) from a small inner 5-fold CV grid INSIDE the
           n budget. May change argmax -- reported, never excluded.
  MetaCal  Ma & Blaschko (ICML 2021) miscoverage-control construction,
           equation-based reimplementation (native semantics): binary
           ranking score (val-fitted logistic on the six logit signals),
           threshold = empirical quantile controlling miscoverage alpha on
           the calibration split, base calibrator = TS-NLL on accepted
           samples. Native outputs: coverage, accepted-set metrics.
  SAG      Semantic-Aware Grouping, OFFICIAL torch code imported from the
           pinned clone (methods/group_calibration.py): linear WNet on
           FEATURES, num_groups=2, num_partitions=20, wd 1e-1, LBFGS,
           combine with TS base (their group_calibration_combine_ts).
All fitting uses only the provided calibration data.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import optimize as sopt
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from srts import softmax as np_softmax, six_signals, T_CLAMP
from maps import fit_ts_nll

EXT = HERE.parent.parent / "external"


# ---------------------------------------------------------------- PTS ------
class _PTSNet(nn.Module):
    def __init__(self, top_k=10, n_nodes=5, n_layers=2):
        super().__init__()
        layers, d = [], top_k
        for _ in range(n_layers):
            layers += [nn.Linear(d, n_nodes), nn.ReLU()]
            d = n_nodes
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.top_k = top_k

    def forward(self, z):  # z: (N, C) raw logits
        zs = torch.sort(z, dim=1, descending=True).values[:, :self.top_k]
        return torch.abs(self.net(zs)).clamp(1e-12, 1e12).squeeze(-1)


class PTS:
    """Official-recipe PTS. objective in {"mse" (native), "bce"}."""

    def __init__(self, objective="mse", steps=100_000, lr=5e-5, wd=0.0,
                 seed=0):
        self.objective = objective
        self.steps, self.lr, self.wd, self.seed = steps, lr, wd, seed
        self.report = {}

    def fit(self, vl, vt):
        torch.manual_seed(self.seed)
        z = torch.tensor(np.clip(np.asarray(vl, dtype=np.float32), -100, 100))
        y = torch.tensor(np.asarray(vt, dtype=np.int64))
        onehot = F.one_hot(y, z.shape[1]).float()
        # official top_k=10 assumes >=10 classes; for C<10 the input is all
        # C sorted logits (documented adaptation, no other change)
        self.model = _PTSNet(top_k=min(10, z.shape[1]))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.wd)
        n = len(y)
        bs = min(1000, n)
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        best = float("inf")
        for s in range(self.steps):
            idx = rng.choice(n, bs, replace=False)
            zb, ob = z[idx], onehot[idx]
            T = self.model(zb)
            P = torch.softmax(zb / T.unsqueeze(-1), dim=1)
            if self.objective == "mse":
                loss = ((P - ob) ** 2).mean()
            else:  # bce on top-label correctness
                corr = (zb.argmax(1) == y[idx]).float()
                pc = P.max(1).values.clamp(1e-7, 1 - 1e-7)
                loss = -(corr * torch.log(pc)
                         + (1 - corr) * torch.log(1 - pc)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 2000 == 0:
                best = min(best, float(loss.item()))
        self.report = {"objective": self.objective, "steps": self.steps,
                       "n_params": sum(p.numel()
                                       for p in self.model.parameters()),
                       "seconds": time.time() - t0, "last_loss_probe": best}
        return self

    @torch.no_grad()
    def predict_proba(self, logits):
        z = torch.tensor(np.clip(np.asarray(logits, dtype=np.float32),
                                 -100, 100))
        T = self.model(z)
        return torch.softmax(z / T.unsqueeze(-1), dim=1).double().numpy()


# ---------------------------------------------------------------- ETS ------
class ETS:
    """Mix-n-Match Ensemble Temperature Scaling (official util code path)."""

    def __init__(self):
        self.report = {}

    def fit(self, vl, vt):
        vl = np.asarray(vl, dtype=np.float64)
        y = np.asarray(vt, dtype=np.int64)
        C = vl.shape[1]
        label = np.eye(C)[y]

        def ll_t(t, logit, lab):
            p = np.clip(np_softmax(logit / t), 1e-20, 1 - 1e-20)
            return -np.sum(lab * np.log(p)) / len(lab)

        bnds = ((0.05, 5.0),)
        t = sopt.minimize(ll_t, 1.0, args=(vl, label), method="L-BFGS-B",
                          bounds=bnds, tol=1e-12)
        self.T = float(t.x)

        p0 = np_softmax(vl / self.T)
        p1 = np_softmax(vl)
        p2 = np.ones_like(p0) / C

        def ll_w(w, *args):
            p = w[0] * p0 + w[1] * p1 + w[2] * p2
            p = np.clip(p, 1e-20, 1 - 1e-20)
            return -np.sum(label * np.log(p)) / len(label)

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
        w = sopt.minimize(ll_w, np.array([1.0, 0.0, 0.0]), method="SLSQP",
                          bounds=((0, 1),) * 3, constraints=cons, tol=1e-12)
        self.w = np.asarray(w.x, dtype=np.float64)
        self.report = {"T": self.T, "w": self.w.tolist(),
                       "converged": bool(w.success)}
        return self

    def predict_proba(self, logits):
        z = np.asarray(logits, dtype=np.float64)
        C = z.shape[1]
        p = (self.w[0] * np_softmax(z / self.T) + self.w[1] * np_softmax(z)
             + self.w[2] / C)
        return p / p.sum(1, keepdims=True)


# ------------------------------------------------------------- Dirichlet ---
class DirichletODIR:
    """Kull et al. 2019 Dirichlet calibration with ODIR regularization."""

    GRID = [(1e-2, 1e-2), (1e-1, 1e-1), (1.0, 1.0), (10.0, 10.0)]

    def __init__(self, seed=0):
        self.seed = seed
        self.report = {}

    @staticmethod
    def _fit_wb(L, y, lam, mu, C):
        # L: log softmax inputs (N,C). Optimize W (C,C), b (C).
        N = len(y)

        def unpack(th):
            return th[:C * C].reshape(C, C), th[C * C:]

        def obj(th):
            W, b = unpack(th)
            S = L @ W.T + b
            S = S - S.max(1, keepdims=True)
            logp = S - np.log(np.exp(S).sum(1, keepdims=True))
            nll = -logp[np.arange(N), y].mean()
            off = W - np.diag(np.diag(W))
            return (nll + lam * (off ** 2).sum() / (C * (C - 1))
                    + mu * (b ** 2).sum() / C)

        th0 = np.concatenate([np.eye(C).ravel(), np.zeros(C)])
        res = minimize(obj, th0, method="L-BFGS-B",
                       options={"maxiter": 500})
        W, b = unpack(res.x)
        return W, b, float(res.fun), bool(res.success)

    def fit(self, vl, vt):
        vl = np.asarray(vl, dtype=np.float64)
        y = np.asarray(vt, dtype=np.int64)
        C = vl.shape[1]
        L = np.log(np.clip(np_softmax(vl), 1e-20, None))
        # inner 5-fold CV over the ODIR grid, ALL within the n budget
        from sklearn.model_selection import StratifiedKFold
        strat = (vl.argmax(1) == y).astype(int)
        folds = min(5, max(2, int(np.bincount(strat).min())))
        best, best_nll = None, float("inf")
        cv_log = []
        if folds >= 2:
            skf = StratifiedKFold(folds, shuffle=True, random_state=self.seed)
            for lam, mu in self.GRID:
                nlls = []
                for tr, te in skf.split(L, strat):
                    W, b, _, _ = self._fit_wb(L[tr], y[tr], lam, mu, C)
                    S = L[te] @ W.T + b
                    S = S - S.max(1, keepdims=True)
                    logp = S - np.log(np.exp(S).sum(1, keepdims=True))
                    nlls.append(-logp[np.arange(len(te)), y[te]].mean())
                m = float(np.mean(nlls))
                cv_log.append({"lam": lam, "mu": mu, "cv_nll": m})
                if m < best_nll:
                    best_nll, best = m, (lam, mu)
        else:
            best = self.GRID[1]
        lam, mu = best
        self.W, self.b, fun, conv = self._fit_wb(L, y, lam, mu, C)
        self.report = {"lam": lam, "mu": mu, "cv": cv_log,
                       "final_obj": fun, "converged": conv}
        return self

    def predict_proba(self, logits):
        z = np.asarray(logits, dtype=np.float64)
        L = np.log(np.clip(np_softmax(z), 1e-20, None))
        S = L @ self.W.T + self.b
        return np_softmax(S)


# --------------------------------------------------------------- MetaCal ---
class MetaCal:
    """Ma & Blaschko 2021 miscoverage-control construction (native
    semantics; equation-based reimplementation, provenance in module doc).

    alpha: target miscoverage (P(accept | wrong) <= alpha, empirically on
    the calibration split). Accepted samples get TS-NLL probabilities;
    rejected samples are abstentions (native output). For the clearly-
    labelled "comparable confidence view" the rejected samples keep the
    base-calibrated probabilities but are flagged.
    """

    def __init__(self, alpha=0.05, seed=0):
        self.alpha = alpha
        self.seed = seed
        self.report = {}

    def fit(self, vl, vt):
        from sklearn.linear_model import LogisticRegression
        vl = np.asarray(vl, dtype=np.float64)
        vt = np.asarray(vt, dtype=np.int64)
        S = six_signals(vl)
        self._mu = S.mean(0)
        self._sd = np.where(S.std(0) < 1e-12, 1.0, S.std(0))
        wrong = (vl.argmax(1) != vt).astype(int)
        if 0 < wrong.sum() < len(wrong):
            clf = LogisticRegression(max_iter=1000, random_state=self.seed)
            clf.fit((S - self._mu) / self._sd, wrong)
            self._clf = clf
            r = clf.predict_proba((S - self._mu) / self._sd)[:, 1]
        else:
            self._clf = None
            r = np.full(len(wrong), wrong.mean())
        # threshold: largest t with empirical miscoverage
        # P(wrong & accepted)/P(wrong) <= alpha  (accept = r <= t)
        if wrong.sum() > 0:
            rw = np.sort(r[wrong == 1])
            k = int(np.floor(self.alpha * len(rw)))
            self.thresh = rw[k - 1] if k >= 1 else -np.inf
        else:
            self.thresh = np.inf
        self.base = fit_ts_nll(vl, vt)
        acc = r <= self.thresh
        self.report = {"alpha": self.alpha, "thresh": float(self.thresh),
                       "cal_coverage": float(acc.mean()),
                       "cal_miscoverage": float(
                           (wrong[acc].sum() / max(1, wrong.sum())))}
        return self

    def _risk(self, logits):
        S = (six_signals(np.asarray(logits, dtype=np.float64))
             - self._mu) / self._sd
        if self._clf is None:
            return np.zeros(len(S))
        return self._clf.predict_proba(S)[:, 1]

    def accept_mask(self, logits):
        return self._risk(logits) <= self.thresh

    def predict_proba(self, logits):
        """Comparable confidence view (flag separately via accept_mask)."""
        return self.base.predict_proba(logits)


# ------------------------------------------------------------------ SAG ----
class SAG:
    """Semantic-Aware Grouping via the OFFICIAL code (pinned clone)."""

    def __init__(self, num_groups=2, num_partitions=20, seed=0):
        self.num_groups = num_groups
        self.num_partitions = num_partitions
        self.seed = seed
        self.report = {}

    def fit(self, vl, vt, features):
        """Official combine-with-TS semantics, all stages within the n
        budget: per random-init partition, WNet+tau trained by soft-mixture
        NLL on (features, logits, labels); deployment per the official
        `calibrate_combine`: HARD group assignment, per-group TS-NLL refit
        on the calibration samples of that group; partition-averaged."""
        sys.path.insert(0, str(EXT / "group_calibration"))
        from methods.group_calibration import optimize_group_fn
        try:
            from omegaconf import OmegaConf
            mkcfg = OmegaConf.create
        except ImportError:  # plain namespace fallback (same attributes)
            import types

            def mkcfg(d):
                def conv(x):
                    if isinstance(x, dict):
                        return types.SimpleNamespace(
                            **{k: conv(v) for k, v in x.items()})
                    return x
                return conv(d)
        feats = torch.tensor(np.asarray(features, dtype=np.float32))
        logits = torch.tensor(np.asarray(vl, dtype=np.float32))
        labels = torch.tensor(np.asarray(vt, dtype=np.int64))
        cfg = mkcfg({"num_groups": self.num_groups,
                     "w_net": {"model": "linear", "weight_decay": 1e-1},
                     "optimizer": {"name": "lbfgs", "lr": 1e-3,
                                   "steps": 100}})
        self._parts = []
        t0 = time.time()
        vl64 = np.asarray(vl, dtype=np.float64)
        vt64 = np.asarray(vt, dtype=np.int64)
        for p in range(self.num_partitions):
            torch.manual_seed(self.seed * 1000 + p)
            tau, w_net = optimize_group_fn(feats, logits, labels,
                                           "linear", False, cfg)
            with torch.no_grad():
                gid = torch.argmax(w_net(feats), dim=1).numpy()
            group_T = []
            pooled = fit_ts_nll(vl64, vt64).temps[0]
            for g in range(self.num_groups):
                m = gid == g
                group_T.append(fit_ts_nll(vl64[m], vt64[m]).temps[0]
                               if m.sum() >= 2 else pooled)
            self._parts.append((w_net, np.asarray(group_T)))
        self.report = {"num_groups": self.num_groups,
                       "num_partitions": self.num_partitions,
                       "seconds": time.time() - t0}
        return self

    @torch.no_grad()
    def predict_proba(self, logits, features):
        z = np.asarray(logits, dtype=np.float64)
        f = torch.tensor(np.asarray(features, dtype=np.float32))
        acc = np.zeros_like(z)
        for w_net, group_T in self._parts:
            gid = torch.argmax(w_net(f), dim=1).numpy()
            T = group_T[gid][:, None]
            acc += np_softmax(z / T)
        p = acc / len(self._parts)
        return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------- AdaTS ----
class _VEnc(nn.Module):
    def __init__(self, z_dim, in_dim, hidden=(1024, 512, 512)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.loc = nn.Linear(d, z_dim)
        self.logs = nn.Linear(d, z_dim)

    def forward(self, x):
        h = self.body(x)
        return self.loc(h), F.softplus(self.logs(h)).clamp(min=1e-3)


class _VDec(nn.Module):
    def __init__(self, z_dim, in_dim, hidden=(1024, 512, 512)):
        super().__init__()
        layers, d = [], z_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers += [nn.Linear(d, in_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class _TPred(nn.Module):
    def __init__(self, n_classes, hidden=(128, 128)):
        super().__init__()
        layers, d = [], n_classes
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, p):
        return self.net(p)


class AdaTS:
    """FAITHFUL port of official AdaTS (thwjoy/adats @ 1fcf1a1,
    calibration.py AdaptiveTemperatureScaling + net.py CondVAE):
    CondVAE on frozen FEATURES; encoder/decoder hidden [1024,512,512];
    class-conditional Gaussian prior; t = softplus(MLP[128,128](per-class
    prior log-prob of sampled z)).clamp(1e-3); loss = ELBO (Laplace recon,
    KL to cond prior) + CE(logits / t, y); Adam 5e-4, 100 epochs, batch 128,
    shuffled. Inference t is stochastic in the official code (z ~ rsample);
    we keep that but fix the torch seed for reproducibility and record it.
    """

    def __init__(self, z_dim=32, seed=0, epochs=100, lr=5e-4, batch=128):
        self.z_dim, self.seed = z_dim, seed
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.report = {}

    def fit(self, vl, vt, features):
        torch.manual_seed(self.seed)
        X = torch.tensor(np.asarray(features, dtype=np.float32))
        Z = torch.tensor(np.asarray(vl, dtype=np.float32))
        Y = torch.tensor(np.asarray(vt, dtype=np.int64))
        n, C = Z.shape
        D = X.shape[1]
        self.C = C
        self.enc = _VEnc(self.z_dim, D)
        self.dec = _VDec(self.z_dim, D)
        self.tp = _TPred(C)
        self.prior_loc = nn.Linear(C, self.z_dim, bias=False)
        self.prior_scale = nn.Linear(C, self.z_dim, bias=False)
        params = (list(self.enc.parameters()) + list(self.dec.parameters())
                  + list(self.tp.parameters())
                  + list(self.prior_loc.parameters())
                  + list(self.prior_scale.parameters()))
        opt = torch.optim.Adam(params, lr=self.lr)
        t0 = time.time()
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                x, z, y = X[idx], Z[idx], Y[idx]
                loc_q, sc_q = self.enc(x)
                zz = loc_q + sc_q * torch.randn_like(sc_q)
                oh = F.one_hot(y, C).float()
                loc_p = self.prior_loc(oh)
                sc_p = F.softplus(self.prior_scale(oh)).clamp(min=1e-3)
                kl = 0.5 * (2 * sc_p.log() - 2 * sc_q.log()
                            + (loc_q - loc_p) ** 2 / sc_p ** 2
                            + sc_q ** 2 / sc_p ** 2 - 1).sum(-1)
                recon = self.dec(zz)
                log_pxz = -(recon - x).abs().sum(-1)  # Laplace scale 1
                elbo_loss = -(log_pxz - kl).mean()
                t = self._t_from(x)
                tce = F.cross_entropy(z / t.view(-1, 1), y)
                loss = elbo_loss + tce
                opt.zero_grad(); loss.backward(); opt.step()
        self.report = {"epochs": self.epochs, "z_dim": self.z_dim,
                       "n_params": sum(p.numel() for p in params),
                       "seconds": time.time() - t0}
        return self

    def _t_from(self, x):
        loc_q, sc_q = self.enc(x)
        zz = (loc_q + sc_q * torch.randn_like(sc_q)
              ).unsqueeze(1).expand(-1, self.C, -1)
        ys = torch.arange(self.C)
        oh = F.one_hot(ys, self.C).float()
        loc_p = self.prior_loc(oh).unsqueeze(0)
        sc_p = F.softplus(self.prior_scale(oh)).clamp(min=1e-3).unsqueeze(0)
        logp = (-0.5 * ((zz - loc_p) / sc_p) ** 2 - sc_p.log()
                - 0.5 * np.log(2 * np.pi)).sum(-1)
        return F.softplus(self.tp(logp)).clamp(min=1e-3).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, logits, features):
        torch.manual_seed(self.seed + 12345)
        x = torch.tensor(np.asarray(features, dtype=np.float32))
        z = torch.tensor(np.asarray(logits, dtype=np.float32))
        t = self._t_from(x)
        return torch.softmax(z / t.view(-1, 1), dim=1).double().numpy()
