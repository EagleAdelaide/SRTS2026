"""Canonical metric registry for the WWW 2027 SRTS-BCE revision.

Single source of truth for every metric reported in the paper. All calibration
errors are returned in PERCENTAGE POINTS (x100) unless the name ends in
``_raw``. Every function takes (probs[N,C], targets[N]) in float64.

Registry versioning: REGISTRY_VERSION bumps on any change in any metric
definition; every result row stores it.

Definitions (paper Appendix, corrected):
  ece          equal-width binned top-confidence ECE (bins param, default 15)
  adaece       equal-mass binned top-confidence ECE
  cw_ece_ovr   one-vs-rest classwise ECE (Kull et al. 2019): for each class c,
               bin p(c|x) over ALL samples against 1[y==c], average over C.
               This is the definition an earlier appendix promised.
  predclass_top_ece
               legacy metric previously (mis)labelled "CW-ECE": per
               PREDICTED-class top-confidence ECE averaged over predicted
               classes. Kept only under this honest name.
  smece        official smooth ECE via the relplot package (Blasiok &
               Nakkiran 2024 released estimator; reflected kernel,
               self-consistent bandwidth). No local reimplementation.
  topbce       top-label binary cross-entropy (the SRTS fitting objective)
  top_brier    top-label Brier ((c - b)^2 mean)
  nll, brier   multiclass proper scores
  aurc, eaurc  risk-coverage summaries
  auroc        AUROC of confidence for correctness
Operational (Web review-queue) metrics live in ops_metrics().
"""
from __future__ import annotations
import numpy as np

REGISTRY_VERSION = "www27-reg-1.0"
_EPS = 1e-12


def _conf_pred_corr(probs, targets):
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    corr = (pred == targets).astype(np.float64)
    return conf, pred, corr


def ece(probs, targets, bins: int = 15) -> float:
    """Equal-width binned top-confidence ECE, percentage points."""
    conf, _, corr = _conf_pred_corr(probs, targets)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, bins - 1)
    val = 0.0
    n = len(conf)
    for b in range(bins):
        m = idx == b
        if m.any():
            val += abs(conf[m].mean() - corr[m].mean()) * (m.sum() / n)
    return float(val * 100.0)


def adaece(probs, targets, bins: int = 15) -> float:
    """Equal-mass binned top-confidence ECE, percentage points."""
    conf, _, corr = _conf_pred_corr(probs, targets)
    order = np.argsort(conf)
    splits = np.array_split(order, bins)
    n = len(conf)
    val = 0.0
    for s in splits:
        if len(s):
            val += abs(conf[s].mean() - corr[s].mean()) * (len(s) / n)
    return float(val * 100.0)


def cw_ece_ovr(probs, targets, bins: int = 15) -> float:
    """One-vs-rest classwise ECE (Kull et al. 2019), percentage points.

    For every class c: bin p(c|x) over ALL samples, compare bin-mean p(c|x)
    with bin empirical frequency of y==c; weight by bin mass; average over C.
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n, C = probs.shape
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for c in range(C):
        pc = probs[:, c]
        yc = (targets == c).astype(np.float64)
        idx = np.clip(np.digitize(pc, edges[1:-1], right=False), 0, bins - 1)
        v = 0.0
        for b in range(bins):
            m = idx == b
            if m.any():
                v += abs(pc[m].mean() - yc[m].mean()) * (m.sum() / n)
        total += v
    return float(total / C * 100.0)


def predclass_top_ece(probs, targets, bins: int = 15) -> float:
    """LEGACY: per-predicted-class top-confidence ECE, averaged over predicted
    classes. This is what earlier code computed under the name
    'classwise_ece15'. Kept for provenance comparisons only."""
    conf, pred, corr = _conf_pred_corr(probs, targets)
    edges = np.linspace(0.0, 1.0, bins + 1)
    vals = []
    for c in np.unique(pred):
        m_c = pred == c
        cc, rr = conf[m_c], corr[m_c]
        idx = np.clip(np.digitize(cc, edges[1:-1], right=False), 0, bins - 1)
        v = 0.0
        for b in range(bins):
            m = idx == b
            if m.any():
                v += abs(cc[m].mean() - rr[m].mean()) * (m.sum() / len(cc))
        vals.append(v)
    return float(np.mean(vals) * 100.0) if vals else 0.0


def smece(probs, targets) -> float:
    """Official smooth ECE (relplot package), percentage points."""
    import relplot  # official implementation; pinned in environment lock
    conf, _, corr = _conf_pred_corr(probs, targets)
    return float(relplot.smECE(conf, corr) * 100.0)


def topbce(probs, targets) -> float:
    conf, _, corr = _conf_pred_corr(probs, targets)
    c = np.clip(conf, _EPS, 1 - _EPS)
    return float(-(corr * np.log(c) + (1 - corr) * np.log(1 - c)).mean())


def top_brier(probs, targets) -> float:
    conf, _, corr = _conf_pred_corr(probs, targets)
    return float(((conf - corr) ** 2).mean())


def nll(probs, targets) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    return float(-np.log(np.clip(probs[np.arange(len(targets)), targets],
                                 _EPS, 1.0)).mean())


def brier(probs, targets) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(targets)), targets] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def top1(probs, targets) -> float:
    _, pred, corr = _conf_pred_corr(probs, targets)
    return float(corr.mean() * 100.0)


def aurc_eaurc(probs, targets):
    conf, _, corr = _conf_pred_corr(probs, targets)
    err = 1.0 - corr
    order = np.argsort(-conf, kind="stable")
    risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    aurc = float(risk.mean())
    risk_opt = np.cumsum(np.sort(err)) / np.arange(1, len(err) + 1)
    return aurc, float(aurc - risk_opt.mean())


def auroc_correctness(probs, targets) -> float:
    conf, _, corr = _conf_pred_corr(probs, targets)
    pos, neg = conf[corr == 1], conf[corr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort(kind="stable").argsort(kind="stable").astype(np.float64)
    # midranks for ties
    order = np.argsort(allv, kind="stable")
    sv = allv[order]
    mid = np.empty_like(ranks)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        mid[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    r_pos = mid[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) - 1) / 2.0)
                 / (len(pos) * len(neg)))


def all_metrics(probs, targets, bins: int = 15, with_smece: bool = True) -> dict:
    aurc, eaurc = aurc_eaurc(probs, targets)
    m = {
        "registry": REGISTRY_VERSION,
        "top1": top1(probs, targets),
        "nll": nll(probs, targets),
        "brier": brier(probs, targets),
        "topbce": topbce(probs, targets),
        "top_brier": top_brier(probs, targets),
        f"ece{bins}": ece(probs, targets, bins),
        f"adaece{bins}": adaece(probs, targets, bins),
        f"cw_ece_ovr{bins}": cw_ece_ovr(probs, targets, bins),
        f"predclass_top_ece{bins}": predclass_top_ece(probs, targets, bins),
        "aurc": aurc, "eaurc": eaurc,
        "auroc": auroc_correctness(probs, targets),
    }
    if with_smece:
        m["smece"] = smece(probs, targets)
    return m


# ---------------------------------------------------------------------------
# Operational (Web review-queue) metrics
# ---------------------------------------------------------------------------

def ops_metrics(probs, targets, alphas=(0.80, 0.90, 0.95)) -> dict:
    """Frozen operational endpoints for the Web moderation-queue protocol.

    For each confidence threshold alpha: auto-accept set A = {conf >= alpha}.
      coverage          |A|/N
      acc_err           actual error rate within A
      pred_err          mean predicted error (1-conf) within A
      err_gap           acc_err - pred_err (signed; + = underestimates errors)
      n_misaccepted     count of wrong predictions inside A
    Plus batch error-count estimation over the whole set:
      pred_err_count    sum(1-conf);  act_err_count  sum(1-b);
      count_rel_err     (pred-act)/act (signed).
    """
    conf, _, corr = _conf_pred_corr(probs, targets)
    out = {}
    N = len(conf)
    for a in alphas:
        A = conf >= a
        k = f"a{int(round(a * 100))}"
        if A.any():
            out[f"cov_{k}"] = float(A.mean())
            out[f"accerr_{k}"] = float(1.0 - corr[A].mean())
            out[f"prederr_{k}"] = float((1.0 - conf[A]).mean())
            out[f"errgap_{k}"] = out[f"accerr_{k}"] - out[f"prederr_{k}"]
            out[f"nmis_{k}"] = int((A & (corr == 0)).sum())
        else:
            out[f"cov_{k}"] = 0.0
            for f_ in ("accerr", "prederr", "errgap"):
                out[f"{f_}_{k}"] = float("nan")
            out[f"nmis_{k}"] = 0
    pec, aec = float((1.0 - conf).sum()), float((1.0 - corr).sum())
    out["pred_err_count"] = pec
    out["act_err_count"] = aec
    out["count_rel_err"] = (pec - aec) / aec if aec > 0 else float("nan")
    out["n"] = N
    return out


def risk_at_coverage(probs, targets, coverages=(0.80, 0.90, 0.95)) -> dict:
    """Selective risk at fixed coverage (ties broken by stable sort)."""
    conf, _, corr = _conf_pred_corr(probs, targets)
    err = 1.0 - corr
    order = np.argsort(-conf, kind="stable")
    out = {}
    for cv in coverages:
        k = max(1, int(round(cv * len(conf))))
        out[f"risk_cov{int(round(cv * 100))}"] = float(err[order[:k]].mean())
    return out
