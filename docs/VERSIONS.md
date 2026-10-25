# VERSIONS

* This package: r3 of 2026-10-16 (paper source commit in SHA256SUMS of the
  release). It supersedes the r2 package (SHA-256 b265a150...), the r1
  revision package (bb4db078...) and srts_www2027_anon_repro.zip
  (1bd1a4ef...), all kept unchanged as historical packages.
* Final figure version (2026-10-25): Figures 2, 4, 5 and 6 and their captions were revised for
  display only (line styles, lighter intervals, labelled boundary g_B = -D/2, axis terms, legend
  placement, Figure 6 at final size).  Every plotted value, interval, classification and method
  order is identical to the earlier version, and the figure-data JSON files are byte-identical;
  paired_audit/test_editorial.py (test_figure_polish) guards the disclosed display offsets.
* Final P0-P3 pass (2026-10-25): contributions are C1-C3 with a reproducibility
  statement after them; Appendix F lists what the anonymized repository holds;
  wording only (ECE ranking, pre-specified external check, lower mean ECE).
  The anonymized repository is this package without the figure PDFs, which
  `reproduce_layer_B.sh` regenerates.
* Footnote patch (2026-10-25): the code-availability sentence moved from the end
  of contribution C4 to a footnote on page 1 (Introduction); wording unchanged.
* Availability patch (2026-10-23): paper/main.tex links the author's anonymous
  code repository and no longer points to this package for any file; this
  package itself is a local reproduction package, not the linked repository.
* Closing terminology edits (2026-10-23): Figure 4 caption and Conclusion name
  D a change in signed bias; label comparisons scoped to the evaluated audits.
* New in the language revision (2026-10-23; wording only): paper/main.tex
  revised for language (no number, figure or evidence status changed);
  verify_paper_numbers.py and paired_audit/test_editorial.py follow the moved
  abstract sentence and add a language guard test.
* New in the final-readability revision (2026-10-22; existing results only):
  Figures 2, 4 and 5 rendered at final size (8 pt text, 7.5 pt ticks;
  gen_fig2_budget.py, gen_fig_boundary.py, gen_fig_final_audit.py); Figure 4b
  encodes budget by colour and solid shape; paired_audit/test_editorial.py
  (11 tests).  All figure-data JSONs are byte-identical to the previous
  release.
* New in the figure-editorial revision (2026-10-22; existing results only):
  gen_fig1_scopes.py (Figure 1: two evaluation scopes and the TvA-TS vs
  SRTS-BCE reversal, from posthoc_figdata/fig1_reversal.json) and
  gen_fig3_tasks.py (Figure 3: three tasks x two endpoints at n = 2500,
  t = 0.90; writes posthoc_figdata/fig3_tasks.json), both rendered at final
  size; gen_endpoint_fig.py output kept as the supplementary full matrix;
  solid markers in Figures 2, 4 and 5 and the coverage figure (marker fill
  carries meaning only in the supplementary matrix); paired_audit/
  test_editorial.py; verify_paper_numbers.py adds the Figure 3 input and
  attribution checks.  The per-cell coverage figure is in paper/figures/
  but no longer in the PDF.
* New in the final-evidence revision (2026-10-22, plan addendum A11 and the
  frozen CivilComments protocol): paired_audit/final_summarize.py (actual
  distinct labels, census cells and same-policy within-pool and transfer
  regrets from the shipped posthoc_closure/dec_*.csv; layer B),
  paired_audit/partition_sensitivity.py (50 salted reviewer partitions; needs
  the cache), paired_audit/test_semantics.py, gen_fig_final_audit.py (Fig. 5
  and the appendix coverage figure; layer B), results/posthoc_final/ and
  results/posthoc_partition/; code/civil/ and results/civil/ (CivilComments
  external check: prepare, train [GPU], calibrate, analyze, audit; logits,
  per-draw rows, draw-0 caches, audit rows, analysis.json and the GPU timing
  records; docs/CIVILCOMMENTS_FROZEN_PROTOCOL.md); paper/main.tex (comments
  removed) so that verify_paper_numbers.py runs inside the package.
* New in the closure revision (2026-10-22, plan addendum A10):
  paired_audit/pools.py (fixed, exact_matched_within_pool and the historical
  stream_matched_then_restricted accepted sets), estimators E4ac and E8-E11
  (error-centred, Poisson AIPW, LURE), decisions.py (G0-G4: historical and
  corrected guardrails, direct H test, joint (D,H) tests, power-targeted
  allocation), test_closure.py, closure_sim.py and closure_summarize.py
  [need the cache]; gen_fig_boundary.py and gen_fig_closure_audit.py
  [layer B]; results/posthoc_closure/ (per-cell est_/dec_ rows,
  estimation_summary.csv, decision_summary.csv, decision_by_margin.csv,
  decision_by_seed_threshold.csv, summary_closure.json).  The A8/A9
  posthoc_audit/ files are kept unchanged as historical records.
* New in the substantive revision (2026-10-21, plan addenda A8, A9 and
  notes 2-3): code/canonical/mechanism_cache.py [layer A: refits from
  logits, ~1.2 CPU-h; writes the per-item cache, not shipped (850 MB)];
  code/canonical/paired_audit/ (identities.py, estimators.py with E0-E7,
  test_identities.py; run_mechanism.py, reverse_accounting.py,
  simulate_audit.py, merge_audit.py, question_separation.py,
  guardrail_sensitivity.py [all need the cache]; relabel_a8_rows.py
  [one-off repair of the A8 rows' map-name column, verified]; print_numbers.py);
  gen_fig_mechanism.py and gen_fig_audit.py [layer B from the shipped
  JSON]; results/posthoc_mechanism/ (analysis_mechanism.json,
  reverse_accounting.json, mechanism_rows.csv, gate.json) and
  results/posthoc_audit/ (audit_rows.csv, audit_rows_active.csv,
  policy_rows.csv, AUDIT_LABEL_EFFICIENCY.csv, summary.json,
  question_separation.*, guardrail_sensitivity.*).
* New in the final candidate (2026-10-20): gen_fig1_reversal.py and
  gen_fig2_budget.py (plan addendum A7; descriptive figure intervals,
  results/posthoc_figdata/), gen_fig4_signed.py (appendix figure from
  signed_descriptive.json); all figure generators now embed Type 42 fonts.
* New in r3: reviewer_cluster_cache.py (layer A: per-reviewer accepted-set
  sums at n = 2500, t = 0.90; writes results/posthoc_reviewer_cluster/
  cache_s*.npz, shipped) and reviewer_cluster_sensitivity.py (layer B), plan
  addendum A6. This anonymous export neutralizes internal names: anchor
  cell directories are renamed (data/c100_cells/ANCHOR_*), comments are
  reworded and paths to unshipped anchor data renamed; arrays, results and
  configuration values are byte-identical. Two internal provenance scripts
  and their one result (no number in the paper) are not included. The
  access timeline ships as its English translation only; the build aborts
  on any non-English text.
* New in r2: analyze_gamma.py, check_gamma_pairing.py, gen_gamma_fig.py
  (paired seen-vs-unseen contrast, docs/POSTHOC_PLAN_2026-10-16.md
  addendum A5), describe_signed_gamma.py (A5 note 3, descriptive) and
  results/posthoc_gamma/.
* docs/FREEZE_MANIFEST.json: SHA-256 of the UNSANITIZED scripts at the
  freeze commit, as verified by the confirmation gate; historical and unedited
  except that the time-zone label of `frozen_at` is removed in this export (all
  times in docs/ and in the UNBLINDED stamps are the same unlabeled clock).
  The confirmation-path scripts are unchanged since then.
* code/: scripts after path sanitization (private roots -> __SRTS_ROOT__),
  so their hashes differ from the freeze manifest by construction.
  New since the earlier package: gen_capacity_fig.py (split argument,
  fix of a split-mixing error in the Amazon capacity figure),
  tests_capacity_fig.py, run_f2_score_control.py, check_f2_reproduction.py,
  analyze_f2.py, analyze_endpoints.py, analyze_family_capacity.py,
  gen_endpoint_fig.py, gen_revision_tables.py, export_web_logits.py,
  join_web_labels.py.
* results/RETRACTED_multiplicity_sensitivity.json is a retracted computation
  (wrong quantile, fresh bootstrap stream) superseded by
  results/statistical_final_audit.json; kept, renamed, for transparency.
* run_matched_coverage.py is the ungated refit of confirmation and
  development rows (writes results/posthoc_matched/); the gated
  run_confirmation_unblinding.py is the historical script and refuses to
  run again by design (UNBLINDED stamps).
* docs/PACKAGE_MANIFEST.json: SHA-256 of every file as shipped.
