#!/bin/bash
# Layer B: recompute every analysis, table and figure-data file of the paper
# from the shipped per-run rows, then compare with the shipped outputs.
# Run from the package root after `python configure.py <root>`.
#   bash reproduce_layer_B.sh            (statistical_final_audit takes ~10 min)
set -e
PKG=$(pwd)
ROOT=$(cat "$PKG/configured_root.txt")/calattn-repo
A=$ROOT/results/www2027
PY=${PY:-python}
REF=$PKG/_reference_outputs
rm -rf "$REF"; mkdir -p "$REF"
( cd "$PKG" && find results paper -type f \( -name '*.json' -o -name '*.csv' -o -name '*.tex' \) \
    ! -name '*provenance*' -print0 | xargs -0 -I{} cp --parents {} "$REF/" )
cd "$PKG/code/canonical"
# frozen and development analyses (exact commands)
$PY analyze_suite.py $A/suite/c100_vit_s*/c100_vit_core_natural.jsonl --test-split test --out $A/analysis_c100_core_natural_ece15.json
$PY analyze_suite.py $A/suite/amazon_s*/amazon_distilbert_core_natural.jsonl --test-split id_test --out $A/analysis_amazon_id_test_ece15.json
$PY analyze_suite.py $A/suite/amazon_s*/amazon_distilbert_core_natural.jsonl --test-split ood_val --out $A/analysis_amazon_ood_val_ece15.json
$PY analyze_suite.py $A/CONFIRMATION/amazon_distilbert.jsonl --test-split CONFIRMATION --out $A/analysis_CONFIRM_amazon_distilbert_ece15.json
$PY analyze_suite.py $A/CONFIRMATION/yahoo_distilbert.jsonl --test-split CONFIRMATION --out $A/analysis_CONFIRM_yahoo_distilbert_ece15.json
$PY analyze_suite.py $A/suite/yahoo_s*/yahoo_distilbert_core_natural.jsonl --test-split id_test --out $A/analysis_yahoo_idtest_ece15.json
$PY analyze_suite.py $A/suite/amazon_rob_s*/amazon_roberta_core_natural.jsonl --test-split id_test --budgets 250,2500 --out $A/analysis_amazon_roberta_idtest.json
$PY analyze_suite.py $A/CONFIRMATION/amazon_distilbert.jsonl --test-split CONFIRMATION --comparisons HTS-BCE:TvA-TS,HTS-BCE:SRTS-BCE,HTS-BCE:SMART+BCE --out $A/analysis_CONFIRM_amazon_htsbce.json
$PY analyze_suite.py $A/suite/yahoo_s*/yahoo_distilbert_core_natural.jsonl --test-split id_test --comparisons TS-NLL:TvA-TS,HTS-NLL:TvA-TS,HTS-BCE:TvA-TS,Margin-K3:TvA-TS,PWLinear-3:TvA-TS,SMART-SoftECE:TvA-TS,SMART+BCE:TvA-TS,SRTS-BCE:TvA-TS --out $A/analysis_yahoo_vs_tva_ece15.json
$PY statistical_final_audit.py
# post-hoc analyses (docs/POSTHOC_PLAN_2026-10-16.md)
$PY analyze_endpoints.py
$PY analyze_endpoints.py development
$PY analyze_f2.py
$PY analyze_family_capacity.py
$PY seed_level_sensitivity.py
$PY analyze_matched.py
$PY analyze_gamma.py   # uses the shipped posthoc_gamma/pairing_check.json (layer A: check_gamma_pairing.py)
$PY describe_signed_gamma.py   # plan A5 note 3, descriptive
$PY reviewer_cluster_sensitivity.py   # plan A6; uses shipped posthoc_reviewer_cluster/cache_s*.npz (layer A: reviewer_cluster_cache.py)
$PY summarize_shift.py
# tables and figure data
$PY gen_main_matrix.py c100_vit test
$PY gen_main_matrix.py amazon_distilbert id_test ece15 "results/www2027/suite/amazon_s*/amazon_distilbert_core_natural.jsonl"
$PY gen_main_matrix.py yahoo_distilbert id_test ece15 "results/www2027/suite/yahoo_s*/yahoo_distilbert_core_natural.jsonl"
$PY gen_ext_tables.py
$PY gen_revision_tables.py
$PY gen_capacity_fig.py --cell c100_vit --split test
$PY gen_capacity_fig.py --cell amazon_distilbert --split id_test
$PY gen_endpoint_fig.py
$PY gen_gamma_fig.py
$PY gen_fig1_reversal.py   # plan A7(i), descriptive
$PY gen_fig1_scopes.py     # Fig. 1 (figure-editorial revision): reads posthoc_figdata/fig1_reversal.json
$PY gen_fig2_budget.py     # plan A7(ii), descriptive; gate vs frozen analyses
$PY gen_fig4_signed.py
$PY gen_fig_boundary.py       # Fig. 4 (plans A8 ii, A9 note 3)
$PY gen_fig_closure_audit.py  # historical Fig. 5 of the closure candidate (plan A10)
$PY paired_audit/final_summarize.py   # plan A11 (i): actual labels, census cells, same-policy regrets
$PY gen_fig_final_audit.py   # Fig. 5 and appendix coverage figure (plan A11); partition rows: layer A (needs the cache)
( cd ../civil && $PY civil_analyze.py && $PY civil_summarize.py )   # CivilComments: endpoint analysis (shipped rows), audit summary (shipped audit rows)
$PY gen_civil_table.py   # appendix table (paper/generated/civil_table.tex)
$PY gen_fig3_tasks.py   # Fig. 3 (figure-editorial revision): writes posthoc_figdata/fig3_tasks.json
# tests and gates (the two gates report the documented PWLinear-3 drift)
$PY tests_capacity_fig.py
$PY check_f2_reproduction.py || echo "F2 gate: see POSTHOC_PLAN Addendum A2 (documented PWLinear-3 drift)"
$PY check_matched_reproduction.py || echo "matched gate: documented PWLinear-3 drift (<= 3.1e-3 ECE points)"
# compare
cd "$PKG"
n=0; bad=0
while IFS= read -r -d '' f; do
  rel=${f#$REF/}; n=$((n+1))
  cmp -s "$f" "$PKG/$rel" || { echo "DIFFERS: $rel"; bad=$((bad+1)); }
done < <(find "$REF" -type f -print0)
echo "layer B: $n outputs compared, $bad differ"
