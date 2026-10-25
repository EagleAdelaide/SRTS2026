#!/usr/bin/env python3
"""WWW2027 WP2: train a text backbone on WILDS Amazon (official task:
review text -> 1..5 stars; official splits; domain = reviewer).

Faithful to the official WILDS ERM config for 'amazon'
(examples/configs/datasets.py @ p-lambda/wilds main, fetched 2026-10-09):
  distilbert-base-uncased, max_token_length 512, cross-entropy,
  batch_size 8, lr 1e-5, weight_decay 0.01, n_epochs 3, AdamW,
  linear decay schedule (no warmup), grad clip 1.0.

DEVIATION (declared, frozen in WEB_PROTOCOL.md): checkpoint selection uses
ID-val accuracy (source-only pipeline), NOT the WILDS-default OOD val, so no
target labels touch any stage. OOD val/test labels are never read here except
to be SAVED alongside dumped logits for later evaluation.

Splits dumped (logits float32 + labels + reviewer ids):
  id_val   -> calibration pool + selection
  id_test  -> source-distribution test
  val      -> OOD development set (disjoint reviewers)
  test     -> OOD confirmation set (disjoint reviewers; untouched until freeze)

Tokenization is cached once per arch under CACHE (memmap int32), shared by
all seeds. Run on a GPU node (no internet needed; HF cache pre-populated).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("__SRTS_ROOT__/data/wilds")
CACHE = Path("__SRTS_ROOT__/data/wilds/tokcache")
OUTBASE = Path("__SRTS_ROOT__/calattn-repo/results/www2027/wilds_amazon")
ARCHS = {"distilbert": "distilbert-base-uncased", "roberta": "roberta-base"}
MAXLEN = 512
N_CLASSES = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tokenize_all(arch: str):
    """Cache token ids/mask for the full WILDS table once per arch."""
    from wilds import get_dataset
    from transformers import AutoTokenizer
    CACHE.mkdir(parents=True, exist_ok=True)
    done = CACHE / f"{arch}.done"
    if done.exists():
        return
    ds = get_dataset(dataset="amazon", download=False, root_dir=str(ROOT))
    # only the official splits we use (the full table has ~4M rows)
    use_splits = ["train", "id_val", "id_test", "val", "test"]
    global_splits = {name: np.where(ds.split_array == ds.split_dict[name])[0]
                     for name in use_splits if name in ds.split_dict}
    rows = np.unique(np.concatenate(list(global_splits.values())))
    pos = {int(g): i for i, g in enumerate(rows)}  # global row -> cache row
    n = len(rows)
    tok = AutoTokenizer.from_pretrained(ARCHS[arch])
    ids = np.lib.format.open_memmap(CACHE / f"{arch}_ids.npy", mode="w+",
                                    dtype=np.int32, shape=(n, MAXLEN))
    msk = np.lib.format.open_memmap(CACHE / f"{arch}_mask.npy", mode="w+",
                                    dtype=np.int8, shape=(n, MAXLEN))
    B = 2048
    texts = ds._input_array  # list[str] of review text (official input)
    for i in range(0, n, B):
        chunk = [str(texts[g]) for g in rows[i:i + B]]
        enc = tok(chunk, truncation=True, max_length=MAXLEN,
                  padding="max_length", return_tensors="np")
        ids[i:i + B] = enc["input_ids"].astype(np.int32)
        msk[i:i + B] = enc["attention_mask"].astype(np.int8)
        if (i // B) % 50 == 0:
            log(f"tokenize {arch}: {i}/{n}")
    ids.flush(); msk.flush()
    # labels + reviewer metadata + split masks, all in CACHE-row coordinates;
    # global row ids kept for provenance
    y = ds.y_array.numpy().astype(np.int64)[rows]
    reviewer = ds.metadata_array[:, 0].numpy().astype(np.int64)[rows]
    splits = {name: np.array([pos[int(g)] for g in gidx], dtype=np.int64)
              for name, gidx in global_splits.items()}
    np.savez(CACHE / f"{arch}_meta.npz", y=y, reviewer=reviewer,
             global_rows=rows,
             **{f"split_{k}": v for k, v in splits.items()})
    done.write_text("ok")
    log(f"tokenize {arch}: DONE n={n}")


def loaders(arch, split_idx, batch, shuffle, seed=0):
    ids = np.load(CACHE / f"{arch}_ids.npy", mmap_mode="r")
    msk = np.load(CACHE / f"{arch}_mask.npy", mmap_mode="r")
    meta = np.load(CACHE / f"{arch}_meta.npz")
    y = meta["y"]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(split_idx)

        def __getitem__(self, i):
            j = split_idx[i]
            return (torch.tensor(np.asarray(ids[j]), dtype=torch.long),
                    torch.tensor(np.asarray(msk[j]), dtype=torch.long),
                    int(y[j]), int(j))

    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(DS(), batch_size=batch, shuffle=shuffle, generator=g,
                      num_workers=4, pin_memory=True, drop_last=False)


@torch.no_grad()
def evaluate(model, loader, device, dump=None, dump_features=False):
    model.eval()
    correct = tot = 0
    outs = []
    for ids_, msk_, y_, j_ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            res = model(input_ids=ids_.to(device),
                        attention_mask=msk_.to(device),
                        output_hidden_states=dump_features)
        z = res.logits.float()
        correct += (z.argmax(1).cpu() == y_).sum().item()
        tot += len(y_)
        if dump is not None:
            feat = None
            if dump_features:
                # mean-pooled last hidden state over attention mask (for the
                # Semantic-Aware Grouping baseline, which needs features)
                h = res.hidden_states[-1].float()
                m = msk_.to(device).unsqueeze(-1).float()
                feat = ((h * m).sum(1) / m.sum(1).clamp_min(1)).cpu().numpy()
            outs.append((z.cpu().numpy(), y_.numpy(), j_.numpy(), feat))
    acc = correct / tot
    if dump is not None:
        Z = np.concatenate([o[0] for o in outs]).astype(np.float32)
        Y = np.concatenate([o[1] for o in outs]).astype(np.int64)
        J = np.concatenate([o[2] for o in outs]).astype(np.int64)
        payload = dict(logits=Z, targets=Y, row_idx=J)
        if dump_features:
            payload["features"] = np.concatenate(
                [o[3] for o in outs]).astype(np.float16)
        np.savez_compressed(dump, **payload)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arch", default="distilbert", choices=list(ARCHS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--probe_steps", type=int, default=0,
                    help="timing probe: train this many steps then exit")
    args = ap.parse_args()

    tokenize_all(args.arch)
    meta = np.load(CACHE / f"{args.arch}_meta.npz")
    sp = {k[6:]: meta[k] for k in meta.files if k.startswith("split_")}
    log(f"splits: { {k: len(v) for k, v in sp.items()} }")

    out = OUTBASE / f"{args.arch}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        ARCHS[args.arch], num_labels=N_CLASSES).to(device)

    tr_loader = loaders(args.arch, sp["train"], args.batch, True, args.seed)
    steps_per_epoch = len(tr_loader)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: max(0.0, 1.0 - s / total_steps))
    scaler = torch.cuda.amp.GradScaler()

    cfg = dict(vars(args), arch_hf=ARCHS[args.arch], maxlen=MAXLEN,
               steps_per_epoch=steps_per_epoch, total_steps=total_steps,
               selection="id_val_accuracy_source_only",
               torch=torch.__version__)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    best_acc, step, t0 = -1.0, 0, time.time()
    for ep in range(args.epochs):
        model.train()
        for ids_, msk_, y_, _ in tr_loader:
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(input_ids=ids_.to(device),
                          attention_mask=msk_.to(device)).logits
                loss = F.cross_entropy(z.float(), y_.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 500 == 0:
                log(f"ep{ep} step {step}/{total_steps} loss {loss.item():.4f} "
                    f"({step / (time.time() - t0):.2f} it/s)")
            if args.probe_steps and step >= args.probe_steps:
                dt = time.time() - t0
                (out / "probe.json").write_text(json.dumps(dict(
                    steps=step, seconds=dt, it_per_s=step / dt,
                    projected_hours=total_steps / (step / dt) / 3600)))
                log(f"PROBE done: {step / dt:.2f} it/s, projected "
                    f"{total_steps / (step / dt) / 3600:.2f} h/seed")
                return
        # source-only selection on id_val
        acc = evaluate(model, loaders(args.arch, sp["id_val"], 64, False), device)
        log(f"epoch {ep}: id_val acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out / "best.pt")
            (out / "best.json").write_text(json.dumps(
                dict(epoch=ep, id_val_acc=acc)))

    # dump logits with the selected checkpoint
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    for split in ["id_val", "id_test", "val", "test"]:
        if split not in sp:
            continue
        acc = evaluate(model, loaders(args.arch, sp[split], 64, False), device,
                       dump=out / f"logits_{split}.npz", dump_features=True)
        log(f"dump {split}: acc {acc:.4f}")
    log("DONE")


if __name__ == "__main__":
    main()
