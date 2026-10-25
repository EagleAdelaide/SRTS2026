#!/usr/bin/env python3
"""CivilComments validation (amended protocol, 2026-10-22): data provenance,
split separation and tokenization.  CPU only.  Reads the pinned Hugging Face
copy google/civil_comments @ f2970eb3a55777454c94069077cc8d9b5866312d.

Test-split LABELS are loaded only to be stored; no test label statistic is
computed or printed here (the test outcome stays unseen until the freeze).

Outputs (DATA_OUT):
  meta.npz       y (toxicity >= 0.5) for every stored row, split index arrays in
                 cache-row coordinates: train_sub, val_check, val_calib, test,
                 test_eval_mask (exact-duplicate exclusions), test_pool (0 audit / 1 evaluation)
  ids.npy, len.npy   distilbert-base-uncased token ids (max 300) and lengths
  provenance.json    counts, duplicate checks, label rates for train/validation only
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REV = "f2970eb3a55777454c94069077cc8d9b5866312d"
SRC = Path(f"__SRTS_ROOT__/data/civil_comments_hf/{REV}/data")
DATA_OUT = Path("__SRTS_ROOT__/data/civil_comments_hf/prepared_v1")
MAXLEN = 300
N_TRAIN = 269038          # WILDS civilcomments train size, as frozen in the 2026-10-20 protocol
SUB_SEED = 20261022       # fixed seed for the training subsample and the validation split
CHECK_FRAC = 0.10
EXPECTED = dict(train=1804874, validation=97320, test=97320)
COLS = ["text", "toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(split):
    files = sorted(SRC.glob(f"{split}-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert list(df.columns) == COLS, (split, list(df.columns))
    assert len(df) == EXPECTED[split], (split, len(df))
    return df


def main():
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = load("train"), load("validation"), load("test")
    prov = dict(source=f"huggingface google/civil_comments @ {REV}", license="cc0-1.0",
                rows=dict(train=len(tr), validation=len(va), test=len(te)),
                label="toxic = toxicity >= 0.5 (fraction of annotators)",
                ids_available=False, identity_columns_available=False, article_ids_available=False)
    for name, df in (("train", tr), ("validation", va), ("test", te)):
        assert df["text"].notna().all() and df["toxicity"].between(0, 1).all(), name
    # exact-duplicate texts (no comment ids exist in this source)
    trs, vas = set(tr["text"]), set(va["text"])
    prov["duplicates"] = dict(
        train_rows_with_text_repeated_in_train=int(tr["text"].duplicated(keep=False).sum()),
        validation_rows_text_in_train=int(va["text"].isin(trs).sum()),
        test_rows_text_in_train=int(te["text"].isin(trs).sum()),
        test_rows_text_in_validation=int(te["text"].isin(vas).sum()),
        test_rows_text_in_train_or_validation=int((te["text"].isin(trs) | te["text"].isin(vas)).sum()))
    # training subsample (fixed seed, all train rows eligible)
    rng = np.random.default_rng(SUB_SEED)
    tr_idx = np.sort(rng.choice(len(tr), size=N_TRAIN, replace=False))
    # validation: drop rows whose text occurs in train; split check / calibration pool
    va_keep = np.where(~va["text"].isin(trs).to_numpy())[0]
    perm = np.random.default_rng(SUB_SEED + 1).permutation(va_keep)
    n_check = int(round(CHECK_FRAC * len(perm)))
    va_check, va_calib = np.sort(perm[:n_check]), np.sort(perm[n_check:])
    # test: evaluation excludes exact duplicates of train or validation texts;
    # audit/evaluation pools by a hash of the row index (no article or user ids exist)
    te_eval = (~(te["text"].isin(trs) | te["text"].isin(vas))).to_numpy()
    te_pool = np.array([int(hashlib.sha256(f"civil-test-{i}".encode()).hexdigest()[:8], 16) % 2 for i in range(len(te))], dtype=np.int8)
    # label rates for train and validation only (not test)
    prov["label_rate"] = dict(train_sub=float((tr["toxicity"].to_numpy()[tr_idx] >= 0.5).mean()),
                              validation_calib=float((va["toxicity"].to_numpy()[va_calib] >= 0.5).mean()))
    prov["sizes"] = dict(train_sub=len(tr_idx), val_check=len(va_check), val_calib=len(va_calib), test=len(te),
                         test_eval=int(te_eval.sum()), test_pool_audit=int((te_pool == 0).sum()), test_pool_eval=int((te_pool == 1).sum()))
    # cache-row layout: [train_sub | validation (all) | test (all)]
    texts = list(tr["text"].to_numpy()[tr_idx]) + list(va["text"]) + list(te["text"])
    y = np.concatenate([(tr["toxicity"].to_numpy()[tr_idx] >= 0.5), (va["toxicity"].to_numpy() >= 0.5),
                        (te["toxicity"].to_numpy() >= 0.5)]).astype(np.int64)
    off_va, off_te = len(tr_idx), len(tr_idx) + len(va)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    n = len(texts)
    ids = np.lib.format.open_memmap(DATA_OUT / "ids.npy", mode="w+", dtype=np.int32, shape=(n, MAXLEN))
    lens = np.zeros(n, dtype=np.int32)
    B = 4096
    for i in range(0, n, B):
        enc = tok(texts[i:i + B], truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="np")
        ids[i:i + B] = enc["input_ids"].astype(np.int32)
        lens[i:i + B] = enc["attention_mask"].sum(1)
        if (i // B) % 20 == 0:
            log(f"tokenize {i}/{n}")
    ids.flush(); np.save(DATA_OUT / "len.npy", lens)
    np.savez(DATA_OUT / "meta.npz", y=y,
             train_sub=np.arange(len(tr_idx)), val_check=off_va + va_check, val_calib=off_va + va_calib,
             test=off_te + np.arange(len(te)), test_eval_mask=te_eval, test_pool=te_pool,
             train_source_rows=tr_idx, val_source_rows=np.arange(len(va)), test_source_rows=np.arange(len(te)))
    prov["tokenizer"] = dict(name="distilbert-base-uncased", max_length=MAXLEN,
                             truncated_share_train_sub=float((lens[:len(tr_idx)] >= MAXLEN).mean()))
    (DATA_OUT / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    log(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
