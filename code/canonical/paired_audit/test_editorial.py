"""Figure-editorial revision (2026-10-22): guards for the targeted wording and encoding fixes.
Run: python -m unittest paired_audit.test_editorial   (from scripts/canonical)
"""
import re
import unittest
from pathlib import Path

REPO = Path("__SRTS_ROOT__/calattn-repo")   # rewritten by the artifact's configure.py, like every other script
TEX = re.sub(r"\s+", " ", (REPO / "www2027/paper/main.tex").read_text())
CAN = Path(__file__).resolve().parents[1]
ABSTRACT = TEX[TEX.index("\\begin{abstract}"):TEX.index("\\end{abstract}")]


class EditorialFixes(unittest.TestCase):
    def test_post_hoc_scope_not_global(self):
        self.assertNotIn("everything else is post hoc", TEX)
        self.assertIn("For Amazon and Yahoo the frozen protocol targeted", TEX)
        self.assertIn("the CivilComments check has its own primary family, fixed before outcomes, with one disclosed amendment", TEX)

    def test_power_084_attributed_to_amazon(self):
        sent = [x for x in re.split(r"(?<=\.)\s", ABSTRACT) if "0.84" in x]
        self.assertEqual(len(sent), 1)
        self.assertIn("Amazon", sent[0]); self.assertIn("0.59", sent[0]); self.assertIn("CivilComments", sent[0])
        self.assertLess(sent[0].index("Amazon"), sent[0].index("0.59"))   # 0.84 attached to Amazon, 0.59 to CivilComments

    def test_score_dominance_scoped_to_amazon(self):
        for phrase in ("most of the change is higher reported confidence", "the score term dominates",
                       "the maps mainly raise the confidence", "changes in reported confidence dominate"):
            for m in re.finditer(re.escape(phrase), TEX):
                self.assertIn("Amazon", TEX[max(0, m.start() - 60):m.start()], phrase)

    def test_general_condition_is_DH(self):
        for m in re.finditer(re.escape("$g_B<-D/2$"), TEX):
            ctx = TEX[max(0, m.start() - 160):m.end() + 40]
            self.assertTrue("DH<0" in ctx or "D>0" in ctx or "positive shift" in ctx, ctx)
        self.assertIn("an accounting consistency rather than an independent prediction", TEX)

    def test_moderation_actions_marked_simulated(self):
        for m in re.finditer(r"automatic removals?", TEX):
            self.assertIn("simulated", TEX[max(0, m.start() - 12):m.start()])
        self.assertNotIn("publication", TEX.replace("publications", ""))

    def test_figure1_wording_and_encoding(self):
        src = (CAN / "gen_fig1_scopes.py").read_text()
        self.assertNotIn("0 = calibrated", src); self.assertIn("0 = zero mean bias", src)
        self.assertNotIn('mfc="white"', src); self.assertIn("auto-process", src); self.assertNotIn("publish", src)

    def test_marker_fill_carries_no_meaning_outside_supplement(self):
        for g in ("gen_fig1_scopes.py", "gen_fig3_tasks.py", "gen_fig2_budget.py", "gen_fig_boundary.py", "gen_fig_final_audit.py"):
            self.assertNotIn('mfc="white"', (CAN / g).read_text(), g)
        self.assertIn("In this figure only, marker fill carries meaning", TEX)

    def test_figure3_keeps_all_methods_and_units(self):
        src = (CAN / "gen_fig3_tasks.py").read_text()
        for m in ("HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"):
            self.assertIn(f'("{m}"', src)
        self.assertIn('100 * c["point"]', src)            # Amazon/Yahoo fractions -> pp, once
        self.assertIn('(c["mean"], c["ci95"][0], c["ci95"][1])', src)   # CivilComments already pp
        self.assertIn("fig:endpoints-full", TEX)             # full matrix retained and referenced

    # --- final-readability revision (2026-10-22)
    def test_targeted_wording_closure(self):
        intro = TEX[TEX.index("\\section{Introduction}"):TEX.index("\\textbf{Notation.}")]
        self.assertIn("toxicity classification as an external check", intro)
        self.assertNotIn("This separates the seen from the unseen Amazon reviewers", TEX)
        self.assertIn("At $n{=}2500$ and $t{=}0.90$ this accounts for the contrasting seen and unseen Amazon directions", TEX)
        self.assertIn("at $t{=}0.80$ the seen-reviewer gap also widens", TEX)
        self.assertIn("strip widths are schematic, not acceptance proportions", TEX)

    def test_coverage_numbers_and_artifact_path_retained(self):
        for x in ("median coverage 0.94 to 0.95", "worst cells reach only 0.83 to 0.89", "median empirical coverage of 0.63",
                  "with a minimum of 0.13 across the evaluated cells", "nominal 95"):
            self.assertIn(x, TEX, x)
        self.assertNotIn("fig\\_coverage.pdf", TEX)   # availability patch: the plot is not in the anonymous repository
        self.assertIn("fig:endpoints-full", TEX)

    def test_main_figures_rendered_at_final_size(self):
        for g in ("gen_fig1_scopes.py", "gen_fig2_budget.py", "gen_fig3_tasks.py", "gen_fig_boundary.py", "gen_fig_final_audit.py"):
            src = (CAN / g).read_text()
            self.assertIn("506.295 / 72.27" if g != "gen_fig1_scopes.py" else "241.14749 / 72.27", src, g)
        for g in ("gen_fig2_budget.py", "gen_fig_boundary.py"):
            self.assertNotIn('bbox_inches="tight"', (CAN / g).read_text(), g)
        self.assertIn('fig.savefig(OUT, metadata={"CreationDate": None})', (CAN / "gen_fig_final_audit.py").read_text())
        src = (CAN / "gen_fig_boundary.py").read_text()
        self.assertIn('(2500, "D", "#0072B2"), (250, "X", "#CC79A7")', src)   # budget: colour and solid shape


    # --- language revision (2026-10-23): residue, self-evaluation, causal and vague wording removed
    def test_language_review(self):
        for gone in ("the earlier Figure 3", "the previous Figure 3", "not itself a contribution",
                     "their measured consequence is the contribution", "We introduce no new calibrator",
                     "cause whole-batch", "We do not claim these phenomena as new", "8 never do", "is cheap",
                     "which has no user shift", "Before any test outcome existed", "times worse", "both beat the",
                     "both interactions survive", "universally superior", "confidence shift and the mean",
                     "Where a confidence shift narrows", "requires many more", "such a shift narrows"):
            self.assertNotIn(gone, TEX, gone)
        self.assertIn("test-set logits but no test labels", TEX)                       # amendment provenance disclosed
        self.assertIn("the change in signed bias and the average of the two signed biases", ABSTRACT)
        self.assertIn("times the RMSE of switcher-stratified sampling", TEX)
        self.assertIn("8.2 times the RMSE of switcher-stratified sampling", TEX)          # closing edit O2
        self.assertIn("Change in signed accepted-set bias $D=g_{\\text{map}}-g_{\\text{scalar}}=S+P$", TEX)   # closing edit T1 (figure-polish term)
        self.assertIn("In the evaluated audits, estimating such a change took far fewer labels", TEX)  # closing edit O1

    # --- availability patch (2026-10-23) and final P0-P3 pass (2026-10-25): exact anonymous URL only;
    # availability claims limited to what the anonymized repository holds; C1-C3 are the contributions
    def test_availability(self):
        url = "https://anonymous.4open.science/r/SRTS2026-EB80/"
        self.assertIn("Our experiments\\footnote{Code is available at \\url{" + url + "}.} measure", TEX)  # page-1 footnote
        self.assertIn("\\paragraph{Code availability.} Code is available at\\\\ \\url{" + url + "}. It holds the "
                      "calibration, audit, analysis and training code, pinned package versions, per-run results, "
                      "number checker and layer-(A) logits; CIFAR-100-C logits are omitted for size, and benchmark "
                      "data and Web labels must be obtained from the original providers.", TEX)
        self.assertEqual(TEX.count(url), 2)
        for gone in ("artifact", "github.com", "\\href{" + url, "claimed to be downloadable",
                     "kept only as a historical record", "\\textbf{Environment.}"):
            self.assertNotIn(gone, TEX, gone)
        self.assertNotIn("supplementary material", TEX.lower())

    # --- figure-polish revision (2026-10-25): display-only changes are disclosed; exact boundary and terms
    def test_figure_polish(self):
        self.assertIn("points are offset horizontally for legibility only", TEX)            # Figure 2 offsets
        self.assertIn("drawn heavier for that reason only", TEX)                          # pre-declared, not preferred
        self.assertIn("Methods are offset within rows for legibility", TEX)               # Figure 6 offsets
        self.assertIn("In this figure only, marker fill carries meaning", TEX)
        fig2 = (CAN / "gen_fig2_budget.py").read_text()
        self.assertIn("off = (k - 1.5) * 0.12", fig2)                                     # offsets unchanged
        b = (CAN / "gen_fig_boundary.py").read_text()
        self.assertIn("a.plot(xs, -2 * xs", b)                                           # exact boundary g_B = -D/2
        self.assertIn("Change in signed accepted-set\\nbias $D$ (pp)", b)
        self.assertIn("Baseline accepted-set bias $g_B$ (pp)", b)
        e = (CAN / "gen_endpoint_fig.py").read_text()
        self.assertIn("H_IN = W_IN * 165.75 / 496.49", e)                                # aspect ratio of the earlier file
        self.assertNotIn('bbox_inches="tight"', e)
        self.assertIn('mfc=col if c["excl0_bonf"] else "white"', e)                       # fill encoding unchanged
        self.assertIn("categorical jitter only", (CAN / "gen_fig_final_audit.py").read_text())

    def test_final_p0_p3(self):
        # P1: reproducibility is a statement after C1-C3, not a fourth contribution
        self.assertNotIn("(C4)", TEX)
        self.assertNotIn("Audit-ready evidence", TEX)
        self.assertIn("(C3) \\emph{Estimation, decision and generalization (RQ3).}", TEX)
        self.assertIn("all main maps preserve top-1 predictions. For reproducibility, the anonymized repository "
                      "provides the code, per-run results and a checker that recomputes every quoted number. "
                      "\\section{Related Work", TEX)
        # P2: the seen-to-unseen transfer statement is about the ECE ranking only
        self.assertIn("so the ECE ranking transferred from seen to unseen reviewers.", TEX)
        self.assertNotIn("so the ranking transferred", TEX)
        # P3-A: CivilComments is a pre-specified external check, not a replication of user shift
        self.assertIn("and run a pre-specified external check on CivilComments.", TEX)
        self.assertNotIn("check the findings on CivilComments", TEX)
        # P3-B: Section 5.1 names the metric
        self.assertIn("the four conditional maps of Table~\\ref{tab:q1} have lower mean ECE than the scalar "
                      "(the native-NLL HTS does not)", TEX)
        self.assertNotIn("beat the scalar", TEX)

if __name__ == "__main__":
    unittest.main()
