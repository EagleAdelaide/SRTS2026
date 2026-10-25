# Accepted-set calibration under label budgets and user shift: review code

Anonymous code for a double-blind submission. It refits every calibrator from the
included logits and recomputes every table, figure, interval and quoted number of
the paper from the included per-run results.

## Layout
| path | contents |
|---|---|
| `code/canonical/` | calibrators, routing and audit simulations, statistical analyses, figure and table generators, the number checker `verify_paper_numbers.py`, tests in `paired_audit/` |
| `code/web/`, `code/civil/` | backbone training and logit export (GPU), CivilComments protocol scripts |
| `data/` | Web logits with official row ids (no labels), CIFAR-100 ViT logits; `data/DOWNLOAD.md` |
| `results/` | per-run result rows (`*.jsonl.gz`, unpacked by `configure.py`), analysis outputs, caches |
| `paper/` | manuscript source, generated tables and figures, as read by the number checker |
| `docs/` | `REPRODUCE.md` (paper item -> script -> inputs), post-hoc analysis plan, CivilComments protocol, access timeline, `VERSIONS.md`, `PACKAGE_MANIFEST.json` (SHA-256 of every file) |
| `env/requirements.lock` | pinned package versions |

## Reproduce on a CPU
```
pip install -r env/requirements.lock --extra-index-url https://download.pytorch.org/whl/cu116
python configure.py /abs/path/to/workdir      # fills the __SRTS_ROOT__ placeholders, links the layout, unpacks the rows
cd code/canonical
python verify_paper_numbers.py                # recomputes every quoted number (79/79 checks)
python -m unittest paired_audit.test_editorial paired_audit.test_semantics paired_audit.test_closure paired_audit.test_identities
cd ../..
OMP_NUM_THREADS=4 bash reproduce_layer_B.sh   # recomputes every table, figure-data file and interval, then compares
```
Layer A (refitting calibrators from logits, after re-joining the Web labels) and
layer C (retraining the backbones) are described in `docs/REPRODUCE.md`. SMART is
bit-exact only with 4 CPU threads.

## Not included
- Amazon (WILDS v2.1) and Yahoo Answers labels and text: obtain them from the
  providers; `code/canonical/join_web_labels.py` re-joins the labels by official
  row id and checks their SHA-256.
- CIFAR-100-C logits (about 1.9 GB): the corruption-shift rows are reproducible
  from the per-run rows only.
- The training pipeline of the CIFAR-100 ViT backbones (their logits are included).

## Figure version
The figures in `paper/figures/` are the final version of the paper. Figures 2, 4, 5 and 6 were revised for
display only (line styles, interval shading, labels, legend placement); every plotted value is unchanged,
and `reproduce_layer_B.sh` regenerates them with the scripts in `code/canonical/` (`docs/VERSIONS.md`).

## Known deviation
`results/statistical_final_audit.json` records the absolute path of the script
that wrote it, so layer B reports one difference, in that field only.
