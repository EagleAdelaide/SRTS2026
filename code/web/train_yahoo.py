#!/usr/bin/env python3
"""WWW2027 WP2 P1-first: Yahoo Answers topic routing (10 classes).

Frozen protocol (WEB_PROTOCOL.md §3): from the official HF
`yahoo_answers_topics` train table (1.4M), carve with RNG(20261009):
  model-train 300,000 / id-val pool 50,000 / id-test 50,000 (disjoint rows);
official test (60,000) = CONFIRMATION (dumped, not evaluated until freeze).
Input text = question_title + " " + question_content (community question as
routed; answers excluded — routing happens before answers exist).
Recipe: DistilBERT, len 256, batch 16, lr 2e-5? -- NO: keep the Amazon family
recipe scaled: batch 16, lr 1e-5, wd 0.01, 2 epochs, AdamW, linear decay,
clip 1.0, fp16. Selection on id-val accuracy (source-only).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CACHE = Path("__SRTS_ROOT__/data/yahoo_tokcache")
OUTBASE = Path("__SRTS_ROOT__/calattn-repo/results/www2027/yahoo")
ARCH = "distilbert-base-uncased"
MAXLEN = 256
N_CLASSES = 10
SPLIT_RNG = 20261009
N_TRAIN, N_IDVAL, N_IDTEST = 300_000, 50_000, 50_000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def tokenize_all():
    done = CACHE / "done"
    if done.exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    d = load_dataset("yahoo_answers_topics")
    tok = AutoTokenizer.from_pretrained(ARCH)

    def text_of(ex):
        return (ex["question_title"] + " " + ex["question_content"]).strip()

    rng = np.random.default_rng(SPLIT_RNG)
    n_train_total = len(d["train"])
    perm = rng.permutation(n_train_total)
    splits = {"train": perm[:N_TRAIN],
              "id_val": perm[N_TRAIN:N_TRAIN + N_IDVAL],
              "id_test": perm[N_TRAIN + N_IDVAL:N_TRAIN + N_IDVAL + N_IDTEST]}

    def encode(tbl, rows, name):
        n = len(rows)
        ids = np.lib.format.open_memmap(CACHE / f"{name}_ids.npy", mode="w+",
                                        dtype=np.int32, shape=(n, MAXLEN))
        msk = np.lib.format.open_memmap(CACHE / f"{name}_mask.npy", mode="w+",
                                        dtype=np.int8, shape=(n, MAXLEN))
        ys = np.zeros(n, dtype=np.int64)
        B = 4096
        for i in range(0, n, B):
            sel = tbl.select([int(x) for x in rows[i:i + B]])
            enc = tok([text_of(e) for e in sel], truncation=True,
                      max_length=MAXLEN, padding="max_length",
                      return_tensors="np")
            ids[i:i + B] = enc["input_ids"].astype(np.int32)
            msk[i:i + B] = enc["attention_mask"].astype(np.int8)
            ys[i:i + B] = np.array(sel["topic"], dtype=np.int64)
            if (i // B) % 20 == 0:
                log(f"tokenize {name}: {i}/{n}")
        ids.flush(); msk.flush()
        np.save(CACHE / f"{name}_y.npy", ys)
        np.save(CACHE / f"{name}_rows.npy", rows)

    for name, rows in splits.items():
        encode(d["train"], rows, name)
    encode(d["test"], np.arange(len(d["test"])), "conf_test")
    json.dump({k: int(len(v)) for k, v in splits.items()},
              open(CACHE / "split_sizes.json", "w"))
    done.write_text("ok")
    log("tokenize DONE")


def loader(name, batch, shuffle, seed=0):
    ids = np.load(CACHE / f"{name}_ids.npy", mmap_mode="r")
    msk = np.load(CACHE / f"{name}_mask.npy", mmap_mode="r")
    y = np.load(CACHE / f"{name}_y.npy")

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(y)

        def __getitem__(self, i):
            return (torch.tensor(np.asarray(ids[i]), dtype=torch.long),
                    torch.tensor(np.asarray(msk[i]), dtype=torch.long),
                    int(y[i]), int(i))

    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(DS(), batch_size=batch, shuffle=shuffle, generator=g,
                      num_workers=4, pin_memory=True)


@torch.no_grad()
def evaluate(model, dl, device, dump=None, dump_features=False):
    model.eval()
    correct = tot = 0
    outs = []
    for ids_, msk_, y_, j_ in dl:
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
                h = res.hidden_states[-1].float()
                m = msk_.to(device).unsqueeze(-1).float()
                feat = ((h * m).sum(1) / m.sum(1).clamp_min(1)).cpu().numpy()
            outs.append((z.cpu().numpy(), y_.numpy(), j_.numpy(), feat))
    if dump is not None:
        payload = dict(
            logits=np.concatenate([o[0] for o in outs]).astype(np.float32),
            targets=np.concatenate([o[1] for o in outs]).astype(np.int64),
            row_idx=np.concatenate([o[2] for o in outs]).astype(np.int64))
        if dump_features:
            payload["features"] = np.concatenate(
                [o[3] for o in outs]).astype(np.float16)
        np.savez_compressed(dump, **payload)
    return correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--probe_steps", type=int, default=0)
    args = ap.parse_args()

    tokenize_all()
    out = OUTBASE / f"distilbert_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        ARCH, num_labels=N_CLASSES).to(device)
    tr = loader("train", args.batch, True, args.seed)
    total = len(tr) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: max(0.0, 1.0 - s / total))
    scaler = torch.cuda.amp.GradScaler()
    (out / "config.json").write_text(json.dumps(
        dict(vars(args), arch=ARCH, maxlen=MAXLEN, total_steps=total,
             selection="id_val_accuracy_source_only", split_rng=SPLIT_RNG)))
    best, step, t0 = -1.0, 0, time.time()
    for ep in range(args.epochs):
        model.train()
        for ids_, msk_, y_, _ in tr:
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(input_ids=ids_.to(device),
                          attention_mask=msk_.to(device)).logits
                loss = F.cross_entropy(z.float(), y_.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 500 == 0:
                log(f"ep{ep} {step}/{total} loss {loss.item():.4f} "
                    f"({step / (time.time() - t0):.2f} it/s)")
            if args.probe_steps and step >= args.probe_steps:
                dt = time.time() - t0
                (out / "probe.json").write_text(json.dumps(dict(
                    steps=step, it_per_s=step / dt,
                    projected_hours=total / (step / dt) / 3600)))
                log(f"PROBE {step/dt:.2f} it/s")
                return
        acc = evaluate(model, loader("id_val", 128, False), device)
        log(f"epoch {ep}: id_val acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out / "best.pt")
            (out / "best.json").write_text(json.dumps(dict(epoch=ep, acc=acc)))
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    for name, fn in [("id_val", "logits_id_val.npz"),
                     ("id_test", "logits_id_test.npz"),
                     ("conf_test", "logits_test.npz")]:
        acc = evaluate(model, loader(name, 128, False), device,
                       dump=out / fn, dump_features=True)
        log(f"dump {name}: acc {acc:.4f}")
    log("DONE")


if __name__ == "__main__":
    main()
