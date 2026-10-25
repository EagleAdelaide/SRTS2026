"""SMART head (logit-gap -> MLP -> temperature), canonical revision.

Topology matches the earlier artifact exactly (1->16->1, ReLU, softplus + eps,
49 trainable parameters, z-scored logit gap input). Fix applied for audit
issue B7: the submitted loop paired the PRE-step loss with the POST-step
weights when snapshotting the best state. Here the state is snapshotted
BEFORE optimizer.step() so (best_loss, best_state) correspond exactly.
`legacy_pairing=True` reproduces the submitted behaviour for the paired
sensitivity comparison.

Objectives: "bce" (top-label BCE; SMART+BCE) and "softece" (original SMART
objective form; see run_smart_softece reference in the earlier repository).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SMARTModel(nn.Module):
    def __init__(self, hidden: int = 16, eps: float = 0.1,
                 margin_mean: float = 0.0, margin_std: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.eps = eps
        self.register_buffer("margin_mean", torch.tensor(float(margin_mean)))
        self.register_buffer("margin_std", torch.tensor(float(margin_std)))

    def forward(self, g):
        x = ((g - self.margin_mean) / self.margin_std).unsqueeze(-1)
        x = F.relu(self.fc1(x))
        return F.softplus(self.fc2(x)).squeeze(-1) + self.eps


def _soft_ece(phat, correct, n_bins: int = 15, temp: float = 0.01):
    """Differentiable soft-binned ECE (original-SMART-style objective)."""
    centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                             dtype=phat.dtype, device=phat.device)
    w = torch.softmax(-((phat[:, None] - centers[None, :]) ** 2) / temp, dim=1)
    mass = w.sum(0) + 1e-12
    conf_b = (w * phat[:, None]).sum(0) / mass
    acc_b = (w * correct[:, None]).sum(0) / mass
    return ((mass / len(phat)) * (conf_b - acc_b).abs()).sum()


def train_smart(logits_val, targets_val, objective: str = "bce",
                n_iter: int = 3000, lr: float = 5e-3, patience: int = 300,
                seed: int = 0, legacy_pairing: bool = False):
    """Returns (model, fit_report)."""
    torch.manual_seed(seed)
    lv = torch.tensor(np.asarray(logits_val), dtype=torch.float32)
    yv = torch.tensor(np.asarray(targets_val), dtype=torch.long)
    Y = (lv.argmax(dim=1) == yv).float()
    top2 = lv.topk(2, dim=1).values
    g = top2[:, 0] - top2[:, 1]
    model = SMARTModel(margin_mean=float(g.mean()),
                       margin_std=float(g.std(unbiased=False).clamp_min(1e-8)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter,
                                                       eta_min=lr * 0.01)
    best_loss, best_state, no_improve, steps = float("inf"), None, 0, 0
    for _ in range(n_iter):
        T = model(g)
        phat = torch.softmax(lv / T.unsqueeze(-1), dim=1).max(dim=1).values
        pc = phat.clamp(1e-7, 1 - 1e-7)
        if objective == "bce":
            loss = -(Y * torch.log(pc) + (1 - Y) * torch.log(1 - pc)).mean()
        elif objective == "softece":
            loss = _soft_ece(pc, Y)
        else:
            raise ValueError(objective)
        lv_f = float(loss.item())
        if not legacy_pairing and lv_f < best_loss - 1e-7:
            # FIX: snapshot the weights that PRODUCED this loss (pre-step)
            best_loss = lv_f
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        steps += 1
        if legacy_pairing:
            # submitted behaviour: compare pre-step loss, save post-step state
            if lv_f < best_loss - 1e-7:
                best_loss = lv_f
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
        else:
            if lv_f >= best_loss - 1e-7:
                no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    return model, {"objective": objective, "best_loss": best_loss,
                   "steps": steps, "n_params": int(n_params),
                   "legacy_pairing": bool(legacy_pairing)}


def smart_predict_proba(model, logits):
    lv = torch.tensor(np.asarray(logits), dtype=torch.float32)
    with torch.no_grad():
        T = model(lv.topk(2, dim=1).values[:, 0] - lv.topk(2, dim=1).values[:, 1])
        P = torch.softmax(lv / T.unsqueeze(-1), dim=1)
    return P.double().numpy()
