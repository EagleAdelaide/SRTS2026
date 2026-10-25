# REPRODUCE — "When Does Conditional Calibration Help?" (WWW 2027 revision)

| layer | what is recomputed | entry point | exactness |
|---|---|---|---|
| A. from logits | refit calibrators, recompute per-run rows | `code/canonical/join_web_labels.py` (once), then `run_suite.py` (development rows), `run_matched_coverage.py` (ungated refit of development and confirmation rows + matched-coverage gaps), `run_f2_score_control.py`, `reviewer_cluster_cache.py` (per-reviewer accepted-set sums for the reviewer-cluster sensitivity, plan A6), `check_gamma_pairing.py` (pairing checks of the seen-vs-unseen contrast: regenerated calibration indices vs stored sampler records, logits/labels vs release), `exploratory_reviewer_level.py`, `exploratory_cluster_bootstrap.py`, `gen_bin_sensitivity.py` | TvA-TS, TS-NLL, HTS, SRTS-BCE bit-exact; SMART bit-exact with 4 CPU threads (OMP/MKL = 4, the scripts' default); L-BFGS-B maps (PWLinear-3, polynomial family, F2 C-variants) drift by up to ~3e-3 ECE points across machines |
| B. from per-run rows | every table, figure-data file and interval in the paper | `bash reproduce_layer_B.sh` (repository root) | byte-identical JSON/TeX/CSV; figures are regenerated as PDFs |
| C. from data | Web backbone training and logit export | `code/web/train_wilds_amazon.py`, `code/web/train_yahoo.py` (GPU) | not bitwise |

Not included / not reproducible from this repository:
- CIFAR-100-C logits (95 cells x 5 seeds, ~1.9 GB): the corruption rows
  (`results/shift`, `results/posthoc_f2/c100c_vit_shift.jsonl`) are layer B only.
- The CIFAR-100 ViT anchor backbones come from the anchor's earlier training
  pipeline, which is not part of this repository; their logits (with labels) are
  included in `data/c100_cells`, so the anchor is layer A from logits.
- The gated `run_confirmation_unblinding.py` is the historical script of the
  single unblinding; it re-verifies the frozen manifest and refuses to run
  once the UNBLINDED stamps exist. Use `run_matched_coverage.py` (writes to
  `results/posthoc_matched/`) to recompute confirmation rows from logits.

## 0. Configure
    python configure.py /abs/path/to/workdir      # rewrites __SRTS_ROOT__, links the layout, decompresses rows
Environment: numpy 1.26.4, scipy 1.13.1, scikit-learn 1.6.1, torch 1.13.1
(CPU), relplot 1.0.3, matplotlib; label join: `wilds==2.0.0`, `datasets`.

## A. Web labels (not redistributed)
    python code/canonical/join_web_labels.py --release data/web_logits \
        --wilds-root /path/to/wilds_data --out <workdir>/calattn-repo/results/www2027
Re-joins labels (and Amazon reviewer ids) from the official WILDS Amazon
v2.1 download and `datasets.load_dataset('yahoo_answers_topics')` by the
shipped official row ids and checks every label vector against its
SHA-256. Draw RNG: 77e6 + 1e6*seed + 1e3*budget + draw.

## B. Paper items -> generators (all run by reproduce_layer_B.sh)

Known exception to byte identity: `results/statistical_final_audit.json` records the absolute path of the script in its `script` field, which differs between hosts; every other output is byte-identical (55 compared on 2026-10-20).
| paper item | generator | inputs |
|---|---|---|
| Figure 1 (two evaluation scopes + illustrative reversal, TvA-TS vs SRTS-BCE) | `gen_fig1_reversal.py` (plan A7, descriptive intervals; writes `posthoc_figdata/fig1_reversal.json` and the earlier figure `fig1_reversal.pdf`) -> `gen_fig1_scopes.py` (`fig1_scopes.pdf`) | CONFIRMATION rows |
| Table 1 (ECE, frozen primaries) | `analyze_suite.py`, `statistical_final_audit.py` -> `gen_revision_tables.py` (`generated/q1_ece_table.tex`) | frozen `analysis_*` JSON, suite / CONFIRMATION rows |
| Figure 2 (budget crossing) | `gen_fig2_budget.py` (plan A7; gate against the frozen analyses) | suite rows, frozen `analysis_*` JSON |
| Figure 3 (three tasks, n=2500, t=0.90: change in count error and in mean absolute accepted-set bias) | `gen_fig3_tasks.py` -> `fig3_tasks.pdf`, `posthoc_figdata/fig3_tasks.json` (Amazon/Yahoo from `posthoc_endpoints/endpoints.json` x100; CivilComments from `civil/analysis.json`, already pp) | as below |
| Appendix Figure 6 (full matrix, the earlier Figure 3: both budgets, three thresholds, matched coverage) | `analyze_endpoints.py` (+ `development`), `analyze_matched.py` -> `gen_endpoint_fig.py` (`endpoints_vs_scalar.pdf`, unchanged) | CONFIRMATION rows, suite rows, `posthoc_matched/*.jsonl` |
| Table 2 (accepted-set contrasts across user populations and splits) | `analyze_endpoints.py`, `analyze_matched.py` -> `gen_revision_tables.py` (`generated/q3_replication_table.tex`) | as Figure 3 |
| Figure 4 (RQ2 boundary $g_B<-D/2$, S/P accounting) | `gen_fig_boundary.py` (layer B) from `posthoc_mechanism/{analysis_mechanism,reverse_accounting}.json`, `mechanism_rows.csv`; to recompute: `mechanism_cache.py` (layer A, ~1.2 CPU-h), `paired_audit/run_mechanism.py`, `paired_audit/reverse_accounting.py` | logits, A4(iii) rows |
| Figure 5 (RQ3 estimate / decide / generalize) and the per-cell coverage figure (not in the paper; `paper/figures/fig_coverage.pdf`) | `gen_fig_final_audit.py` (layer B) from `posthoc_closure/{summary_closure.json, dec_*.csv}`, `posthoc_figdata/fig_closure_coverage.json`, `posthoc_partition/partition_summary.json`; to recompute: `paired_audit/closure_sim.py` (plan A10, ~0.5 h on 20 cores, needs the cache), `paired_audit/closure_summarize.py`, `paired_audit/partition_sensitivity.py` (plan A11 ii, needs the cache) | cache from `mechanism_cache.py` |
| RQ3 label accounting and same-policy regrets (caps vs actual distinct labels, census cells, within-pool vs transfer) | `paired_audit/final_summarize.py` (plan A11 i; layer B) -> `posthoc_final/{decision_same_policy.csv, margin_bins.csv, summary_final.json}` | `posthoc_closure/dec_*.csv` |
| reviewer-partition sensitivity (plan A11 ii; RQ3 "Generalize", Fig. 5c) | `paired_audit/partition_sensitivity.py` -> `posthoc_partition/{partition_rows.csv, partition_summary.json}` (GO rule in `go_rule`) | cache (layer A) |
| CivilComments external check (frozen protocol, amendment F1) | `code/civil/prepare_civil.py` (data; provenance in `docs/civil_provenance.json`), `train_civil.py` + `sbatch_civil_train.sh` (GPU, layer C), `civil_calibrate.py` (layer A from the shipped logits), `civil_analyze.py` (layer B) -> `results/civil/analysis.json`, `civil_audit.py` (from the shipped draw-0 caches) -> `results/civil/audit_{est,dec}_s*.csv` | `results/civil/distilbert_s*/logits_*.npz`, `rows_s*.jsonl`, `cache_s*_n*.npz` |
| App. Table (count error, coverage and accepted-set endpoints by threshold) | `gen_revision_tables.py` (`generated/app_endpoints_table.tex`) | `posthoc_endpoints/endpoints.json` |
| App. Table (external baselines) | `gen_ext_tables.py` (`generated/ext_baselines.tex`) | suite_ext rows, efficiency JSON |
| App. Table (paired seen-vs-unseen contrast, plan A5) | `analyze_gamma.py` (requires `posthoc_gamma/pairing_check.json` from `check_gamma_pairing.py`, layer A) -> `gen_revision_tables.py` (`generated/gamma_rep_table.tex`); signed description: `describe_signed_gamma.py` | `posthoc_matched/*.jsonl`, suite and CONFIRMATION rows |
| reviewer-cluster sensitivity (plan A6) | `reviewer_cluster_sensitivity.py` | `posthoc_reviewer_cluster/cache_s*.npz`, `posthoc_gamma/analysis_gamma.json` |
| within-family and seed-level statements | `analyze_family_capacity.py`, `seed_level_sensitivity.py` | suite rows, analysis outputs |
| corruption-shift numbers | `summarize_shift.py` | `shift/*.jsonl` |
| every number quoted in the paper | `verify_paper_numbers.py` (recomputes each from the result files and checks `paper/main.tex`); `paired_audit/test_semantics.py`, `test_closure.py`, `test_identities.py` | all of the above |
| historical records (superseded, kept) | `gen_fig_closure_audit.py` (closure-candidate Fig. 5), `gen_fig_audit.py`, `paired_audit/simulate_audit.py`, `merge_audit.py`, `guardrail_sensitivity.py` (A8/A9) | kept unchanged |
| post-hoc plan and addenda | `docs/POSTHOC_PLAN_2026-10-16.md` (A1-A11) | — |

## Integrity
`docs/FREEZE_MANIFEST.json` (hashes of the unsanitized scripts at the
freeze commit, verified by the confirmation gate) is a historical record,
unedited except that the time-zone label of its timestamp is removed; `docs/PACKAGE_MANIFEST.json` hashes every file in this repository;
`docs/VERSIONS.md` explains the namespaces; `docs/ACCESS_TIMELINE.md`
(English translation of the internal record) discloses every
access to the confirmation sets.
