#!/usr/bin/env python3
"""CivilComments validation backbone (amended protocol, 2026-10-22).

Recipe = the repository's WILDS-Amazon DistilBERT recipe adapted to binary
toxicity: distilbert-base-uncased, max length 300, batch 8, AdamW lr 1e-5,
weight decay 0.01, linear decay without warmup, gradient clip 1.0, fp16 autocast,
ONE epoch on the fixed 269,038-comment training subsample.  No checkpoint
selection: the end-of-epoch model is used; the epoch-check slice (10% of the
validation split, disjoint from the calibration pool) is only logged.
Batches are trimmed to their longest sequence (attention mask unchanged), which
leaves the computation identical to fixed 300-token padding.

Writes OUT/distilbert_s{seed}/: config.json, timing.json, logits_{val_calib,
val_check,test}.npz (logits float32, targets, row_idx).  Test logits are dumped
with their labels for the later CPU analysis; nothing here computes a test metric.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA = Path("__SRTS_ROOT__/data/civil_comments_hf/prepared_v1")
OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027/civil")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Rows(Dataset):
    def __init__(self, idx, ids, lens, y):
        self.idx, self.ids, self.lens, self.y = idx, ids, lens, y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return np.asarray(self.ids[j]), int(self.lens[j]), int(self.y[j]), j


def collate(batch):
    L = max(1, max(b[1] for b in batch))
    ids = torch.tensor(np.stack([b[0][:L] for b in batch]), dtype=torch.long)
    lens = torch.tensor([b[1] for b in batch])
    mask = (torch.arange(L)[None, :] < lens[:, None]).long()
    return ids, mask, torch.tensor([b[2] for b in batch]), torch.tensor([b[3] for b in batch])


@torch.no_grad()
def dump(model, ds, device, path, use_amp, accuracy=True):
    model.eval()
    Z, Y, J = [], [], []
    for ids_, msk_, y_, j_ in DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            z = model(input_ids=ids_.to(device), attention_mask=msk_.to(device)).logits.float()
        Z.append(z.cpu().numpy()); Y.append(y_.numpy()); J.append(j_.numpy())
    Z, Y, J = np.concatenate(Z).astype(np.float32), np.concatenate(Y), np.concatenate(J)
    np.savez_compressed(path, logits=Z, targets=Y, row_idx=J)
    return float((Z.argmax(1) == Y).mean()) if accuracy else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="CPU implementation check on a tiny subset")
    args = ap.parse_args()
    t_start = time.time()
    meta = np.load(DATA / "meta.npz")
    ids = np.load(DATA / "ids.npy", mmap_mode="r"); lens = np.load(DATA / "len.npy"); y = meta["y"]
    tr, vcheck, vcal, te = meta["train_sub"], meta["val_check"], meta["val_calib"], meta["test"]
    if args.smoke:
        tr, vcheck, vcal, te = tr[:64], vcheck[:32], vcal[:32], te[:32]
    out = OUT / (f"smoke_s{args.seed}" if args.smoke else f"distilbert_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.smoke else "cuda"
    use_amp = device == "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(Rows(tr, ids, lens, y), batch_size=8, shuffle=True, generator=g,
                        collate_fn=collate, num_workers=0 if args.smoke else 4, pin_memory=not args.smoke)
    total = len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / total))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    cfg = dict(seed=args.seed, smoke=args.smoke, epochs=1, batch=8, lr=1e-5, wd=0.01, maxlen=300, steps=total,
               n_train=len(tr), n_val_check=len(vcheck), n_val_calib=len(vcal), n_test=len(te),
               selection="none (end of epoch)", torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if use_amp else "cpu")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    model.train(); t0 = time.time()
    for step, (ids_, msk_, y_, _) in enumerate(loader, start=1):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            z = model(input_ids=ids_.to(device), attention_mask=msk_.to(device)).logits
            loss = F.cross_entropy(z.float(), y_.to(device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 1000 == 0 or step == total:
            log(f"step {step}/{total} loss {loss.item():.4f} ({step / (time.time() - t0):.1f} it/s)")
    t_train = time.time() - t0
    acc_check = dump(model, Rows(vcheck, ids, lens, y), device, out / "logits_val_check.npz", use_amp)
    log(f"epoch-check slice accuracy {acc_check:.4f} (logged only; no selection)")
    dump(model, Rows(vcal, ids, lens, y), device, out / "logits_val_calib.npz", use_amp)
    dump(model, Rows(te, ids, lens, y), device, out / "logits_test.npz", use_amp, accuracy=False)  # no test metric
    (out / "timing.json").write_text(json.dumps(dict(train_seconds=t_train, total_seconds=time.time() - t_start,
                                                     val_check_accuracy=acc_check), indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
