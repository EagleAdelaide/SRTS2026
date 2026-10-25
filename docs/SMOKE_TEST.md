# SMOKE TEST — revision package (2026-10-16, run from a fresh unzip)

Procedure: unzip into an empty directory; `python configure.py <root>`;
documented environment (numpy 1.26.4, scipy 1.13.1, scikit-learn 1.6.1,
torch 1.13.1 CPU, 4 threads).

## Layer B — `bash reproduce_layer_B.sh`
All analysis, table and figure-data outputs were regenerated from the shipped
rows (9 frozen/development analyses, statistical audit, endpoint analyses on
confirmation and development splits, F2, within-family, seed-level,
matched-coverage and shift summaries, all generated table bodies, figure
JSON/CSV) and compared with the shipped copies: 46 files compared, 45
byte-identical; the one difference is the `script` path recorded inside
statistical_final_audit.json (package path instead of the original path),
with every number identical. tests_capacity_fig.py: 4/4 pass. The two
reproduction gates report the documented PWLinear-3 drift (plan addenda A2,
A4): D-OOF exact; PWLinear-3 / C-IN within 3.1e-3 ECE points.

## Layer A — from logits
- `join_web_labels.py`: labels for all 29 Web files re-joined from the
  official WILDS Amazon v2.1 download and Hugging Face
  `yahoo_answers_topics`; every SHA-256 matched; Amazon reviewer ids joined
  (1,334 confirmation reviewers).
- Ungated refit (`run_matched_coverage.fit5`, same path as `run_suite`):
  seed 1, n = 250, draw 3, both confirmation sets, compared with the shipped
  CONFIRMATION rows:
    A amazon_distilbert s1 n250 d3 TvA-TS      |dECE| 0.0e+00 |dgap.90| 0.0e+00
    A amazon_distilbert s1 n250 d3 HTS-BCE     |dECE| 0.0e+00 |dgap.90| 0.0e+00
    A amazon_distilbert s1 n250 d3 SRTS-BCE    |dECE| 0.0e+00 |dgap.90| 0.0e+00
    A amazon_distilbert s1 n250 d3 SMART+BCE   |dECE| 0.0e+00 |dgap.90| 0.0e+00
    A amazon_distilbert s1 n250 d3 PWLinear-3  |dECE| 6.0e-07 |dgap.90| 1.1e-08
    A yahoo_distilbert s1 n250 d3 TvA-TS      |dECE| 0.0e+00 |dgap.90| 0.0e+00
    A yahoo_distilbert s1 n250 d3 HTS-BCE     |dECE| 0.0e+00 |dgap.90| 0.0e+00
    A yahoo_distilbert s1 n250 d3 SRTS-BCE    |dECE| 0.0e+00 |dgap.90| 0.0e+00
    A yahoo_distilbert s1 n250 d3 SMART+BCE   |dECE| 0.0e+00 |dgap.90| 0.0e+00
    A yahoo_distilbert s1 n250 d3 PWLinear-3  |dECE| 9.4e-08 |dgap.90| 8.8e-09
- Earlier in the same session (repository copy, same code): seed 0 draw 0
  at n = 250 and 2500 on both confirmation sets and the anchor, plus the four
  F2 variants: bit-exact except PWLinear-3 / C-variants (<= 3.2e-5).
- Full matched-coverage refit (4,200 rows, all seeds and draws at n = 250
  and 2500): fixed-threshold metrics reproduce the saved rows exactly for
  TvA-TS, HTS+BCE, SRTS-BCE, SMART+BCE; PWLinear-3 within 3.1e-3 (three draws
  above 1e-3).
- Recovered exploratory generators rerun from logits:
  exploratory_reviewer_level.py and exploratory_cluster_bootstrap.py
  reproduce the shipped JSON byte-identically.
  gen_bin_sensitivity.py (full-pool fits) also reproduces bin_sensitivity.json
  byte-identically.

## Not covered
- Corruption-shift rows: CIFAR-100-C logits are not shipped (layer B only).
- Backbone training (layer C).
