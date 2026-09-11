"""
Generate numbers.tex, tab_headline.tex and tab_loo.tex directly from the
result CSVs.

Every quantity quoted in the manuscript is defined here and nowhere else, so
the prose cannot drift away from the data: if a number changes, the paper
changes with it on the next build.

Author: Liang Dong.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PAPER = HERE.parent.parent

SC_ORDER = ["narrowband", "wideband", "bursty", "backhaul",
            "compromised_xapp", "compound"]
SC_TEX = {"narrowband": "Narrowband", "wideband": "Wideband",
          "bursty": "Bursty", "backhaul": "Backhaul",
          "compromised_xapp": "Comp.\\ xApp", "compound": "Compound"}
CT_ORDER = ["static", "heuristic", "structured", "greedy_twin", "llm_only",
            "lagrangian_rl", "simplex_rta", "shield_filter", "shield_repair"]
CT_TEX = {"static": "Static", "heuristic": "Heuristic",
          "structured": "Struct.", "greedy_twin": "Greedy-Twin",
          "llm_only": "LLM-only",
          "lagrangian_rl": "Lagr.-RL", "simplex_rta": "React-RTA",
          "shield_filter": "Shield(filt.)",
          "shield_repair": "\\textbf{Shield(rep.)}"}
PHI_TEX = {"phi_rf": "$\\phi_{\\mathrm{rf}}$", "phi_id": "$\\phi_{\\mathrm{id}}$",
           "phi_qos": "$\\phi_{\\mathrm{qos}}$", "phi_bh": "$\\phi_{\\mathrm{bh}}$",
           "phi_u": "$\\phi_{u}$", "phi_pol": "$\\phi_{\\mathrm{pol}}$",
           "none": "--- (full $\\Phi$)"}

NUM: dict[str, str] = {}


def put(name: str, value: str) -> None:
    NUM[name] = value


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\,\\%"


def num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def pts(x: float, d: int = 1) -> str:
    """A difference of two rates is percentage POINTS.  Rendering it as "%"
    invites the reader to take it as a relative change, which it is not."""
    return f"{100 * x:.{d}f}\\,points"


# ---------------------------------------------------------------------------
master = pd.read_csv(RESULTS / "master.csv")
MAIN = master[(master.planner == "naive") & (master.predicate_set == "phi") &
              (master.perception == "radioml")]
ADAPT = master[(master.planner == "adaptive") & (master.perception == "radioml")]

put("Seeds", str(int(MAIN.seed.nunique())))
put("NumPairs", str(int(MAIN.scenario.nunique() * MAIN.seed.nunique())))
put("Cycles", str(int(MAIN.cycles.iloc[0])))
put("NumEpisodes", f"{len(master):,}".replace(",", "{,}"))

g = MAIN.groupby("controller")
put("PDRShieldRepair", num(g.mean_pdr.mean()["shield_repair"]))
put("PDRShieldRepairP", num(g.mean_pdr.mean()["shield_repair"], 4))
put("PDRGreedyP", num(g.mean_pdr.mean()["greedy_twin"], 4))
put("PDRShieldFilter", num(g.mean_pdr.mean()["shield_filter"]))
put("PDRGreedy", num(g.mean_pdr.mean()["greedy_twin"]))
put("PDRHeuristic", num(g.mean_pdr.mean()["heuristic"]))
put("ViolGreedy", pct(g.violation_rate_phi.mean()["greedy_twin"]))
put("ViolLLMOnly", pct(g.violation_rate_phi.mean()["llm_only"]))
put("ViolLagrangian", pct(g.violation_rate_phi.mean()["lagrangian_rl"]))
put("ViolSimplex", pct(g.violation_rate_phi.mean()["simplex_rta"]))
put("HazShieldRepair", num(g.hazard_rate.mean()["shield_repair"]))
put("FallbackFilter", pct(g.fallback_rate.mean()["shield_filter"]))
put("RecoveryShield", num(g.recovery_cycles.mean()["shield_repair"], 1))
put("RecoveryHeur", num(g.recovery_cycles.mean()["heuristic"], 1))

# --- paired tests ----------------------------------------------------------
piv = MAIN.pivot_table(index=["scenario", "seed"], columns="controller",
                       values="mean_pdr")


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((a[:, None] > b[None, :]).sum() -
                  (a[:, None] < b[None, :]).sum()) / (a.size * b.size))


def holm(ps):
    m, order, adj, run = len(ps), np.argsort(ps), np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


others = [c for c in CT_ORDER if c != "shield_repair"]
raw = []
for c in others:
    a, b = piv["shield_repair"].to_numpy(), piv[c].to_numpy()
    raw.append(1.0 if np.allclose(a, b) else float(sps.wilcoxon(a, b, zero_method="zsplit").pvalue))
adj = holm(raw)
for c, p in zip(others, adj):
    a, b = piv["shield_repair"].to_numpy(), piv[c].to_numpy()
    key = {"shield_filter": "Filter", "heuristic": "Heur",
           "greedy_twin": "Greedy", "llm_only": "LLMOnly",
           "lagrangian_rl": "Lagr", "simplex_rta": "Simplex",
           "structured": "Struct", "static": "Static"}[c]
    put(f"Cliff{key}", f"{cliffs(a, b):+.2f}")
    put(f"Diff{key}", f"{(a - b).mean():+.3f}")
    put(f"Gap{key}", f"{abs((a - b).mean()):.3f}")
    # Emitted WITHOUT math delimiters so the macro can be used inside an
    # existing math group, e.g. $p_{\mathrm{Holm}}\PValGreedy$.
    put(f"PVal{key}", ("<10^{-6}" if p < 1e-6 else
                       f"<10^{{{int(np.ceil(np.log10(p)))}}}" if p < 0.01 else
                       f"={p:.2f}"))

# --- enforcement isolated from the attacker's own retargeting -------------
# In E2 the Kerckhoffs attacker retargets into whatever the defender enforces,
# so under Phi+ it sanitises its own proposals and EVERY controller inherits the
# reduction.  E2b pins the attacker to Phi and varies only the enforced set, so
# the proposal stream is identical to the E2 "phi" arm.
_xp = RESULTS / "adaptive_crossed.csv"
if _xp.exists():
    X = pd.read_csv(_xp).groupby("controller").mean(numeric_only=True)
    _E2 = master[(master.planner == "adaptive")
                 & (master.predicate_set == "phi")].groupby("controller").mean(
                     numeric_only=True)
    # A candidate-restricted selector cannot comply on a cycle whose entire
    # pool is inadmissible, so its violation rate is only partly about the
    # selector.  Report availability and split the violations accordingly.
    _L = pd.read_csv(_xp); _L = _L[_L.controller == "lagrangian_rl"]
    _av, _vi = _L.compliant_available_rate, _L.violation_rate_phi
    _forced = (1 - _av).mean()
    _cond = ((_vi - (1 - _av)) / _av).clip(0, 1).mean()
    put("CrossFilterFallback",
        pct(float(pd.read_csv(_xp).set_index("controller")
                  .loc["shield_filter", "fallback_rate"].mean()), 1))

    def crossed_table() -> str:
        _T = pd.read_csv(_xp).groupby("controller").mean(numeric_only=True)
        _rows = [("greedy_twin", "Greedy-Twin (unshielded)"),
                 ("lagrangian_rl", "Lagrangian-RL"),
                 ("simplex_rta", "Reactive-RTA"),
                 ("shield_filter", "Shield (filter)"),
                 ("shield_repair", "\\textbf{Shield (repair)}")]
        out = ["\\begin{tabular}{lcccc}", "\\toprule",
               "Controller & PDR & Hazard & Violates $\\Phi^{+}$ & Avail. \\\\",
               "\\midrule"]
        for _k, _lab in _rows:
            r = _T.loc[_k]
            out.append(f"{_lab} & {r.mean_pdr:.3f} & {r.hazard_rate:.3f} & "
                       f"{r.violation_rate_phi:.3f} & "
                       f"{r.compliant_available_rate:.3f} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(out)

    (PAPER / "tab_crossed.tex").write_text(crossed_table() + "\n")
    put("CrossAvail", pct(_av.mean(), 1))
    put("CrossForced", pct(_forced, 1))
    put("CrossCondViol", pct(_cond, 1))
    _availsp = pd.read_csv(_xp).groupby("controller").compliant_available_rate.mean()
    for _c, _k in (("greedy_twin", "Unshield"), ("lagrangian_rl", "Lagr"),
                   ("simplex_rta", "RTA"), ("shield_filter", "Filter"),
                   ("shield_repair", "Repair")):
        put(f"Cross{_k}Haz", num(X.loc[_c, "hazard_rate"]))
        put(f"Cross{_k}Viol", pct(X.loc[_c, "violation_rate_phi"], 1))
        put(f"Cross{_k}PDR", num(X.loc[_c, "mean_pdr"]))
    # Held-fixed proposals: the unshielded optimiser does not move.
else:
    for _k in ("Unshield", "Lagr", "RTA", "Filter", "Repair"):
        for _q in ("Haz", "Viol", "PDR"):
            put(f"Cross{_k}{_q}", "n/a")

# --- risk--PDR frontier: threshold tuning vs predicate expansion ----------
_fp = RESULTS / "frontier.csv"
if _fp.exists():
    F = pd.read_csv(_fp)
    _R = F[F.controller == "shield_repair"].groupby(
        ["predicate_set", "ceiling_dbm"]).mean(numeric_only=True)
    _S = F[F.controller == "structured"].groupby(
        ["predicate_set", "ceiling_dbm"]).mean(numeric_only=True)

    def _r(c, ps="phi"):
        return _R.loc[(ps, c)]

    # LaTeX macro names cannot contain digits, so the ceilings are named
    # high/mid/low and emitted as macros rather than typed into the prose.
    _HI, _MID, _LO = sorted(c for c in F.ceiling_dbm.unique() if c < 30.0)[::-1]
    put("FrontCeilHi", f"{_HI:.0f}")
    put("FrontCeilMid", f"{_MID:.0f}")
    put("FrontCeilLo", f"{_LO:.0f}")
    put("FrontPDRDropStep", pts(float(_r(30).mean_pdr - _r(_HI).mean_pdr), 1))
    put("FrontHazHi", num(float(_r(_HI).hazard_rate)))
    put("FrontPDRHi", num(float(_r(_HI).mean_pdr)))
    put("FrontHazMid", num(float(_r(_MID).hazard_rate)))
    put("FrontPDRMid", num(float(_r(_MID).mean_pdr)))
    put("FrontHazLo", num(float(_r(_LO).hazard_rate)))
    put("FrontPDRLo", num(float(_r(_LO).mean_pdr)))
    put("FrontStarveLo", num(float(_r(_LO).h_starve), 4))
    put("FrontRelayFloor", num(float(_r(_MID).h_relay)))
    put("FrontStructHazHi", num(float(_S.loc[("phi", _HI)].hazard_rate)))
    put("FrontStructPDRHi", num(float(_S.loc[("phi", _HI)].mean_pdr)))
    put("FrontStructDropHi",
        pts(float(_S.loc[("phi", 30.0)].mean_pdr
                  - _S.loc[("phi", _HI)].mean_pdr), 1))
    put("FrontPlusOverTuned",
        pts(float(_r(_MID).mean_pdr - _r(30, "phi_plus").mean_pdr), 1))

    def frontier_table() -> str:
        rows = [("phi", c, "$\\Phi$, %.0f\\,dBm%s"
                 % (c, "$^{\\dagger}$" if c == 30.0 else ""))
                for c in (30.0, _HI, _MID, _LO)]
        rows.append(("phi_plus", 30.0, "$\\Phi^{+}$, 30\\,dBm"))
        out = ["\\setlength{\\tabcolsep}{3.4pt}",
               "\\begin{tabular}{lccccc}", "\\toprule",
               "Policy & PDR & Haz. & $h_{\\mathrm{therm}}$ & "
               "$h_{\\mathrm{pwr}}$ & $h_{\\mathrm{relay}}$ \\\\",
               "\\midrule"]
        for ps, c, lab in rows:
            r = _R.loc[(ps, c)]
            bold = "\\textbf{%s}" % lab if ps == "phi_plus" else lab
            out.append(f"{bold} & {r.mean_pdr:.3f} & {r.hazard_rate:.3f} & "
                       f"{r.h_therm:.3f} & {r.h_pwr:.3f} & {r.h_relay:.3f} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(out)

    (PAPER / "tab_frontier.tex").write_text(frontier_table() + "\n")
else:
    for _k in ("FrontCeilHi", "FrontCeilMid", "FrontCeilLo", "FrontPDRDropStep",
               "FrontHazHi", "FrontPDRHi", "FrontHazMid", "FrontPDRMid",
               "FrontHazLo", "FrontPDRLo", "FrontStarveLo", "FrontRelayFloor",
               "FrontStructHazHi", "FrontStructPDRHi", "FrontStructDropHi",
               "FrontPlusOverTuned"):
        put(_k, "n/a")

# --- adaptive adversary / coverage gap ------------------------------------
ap = ADAPT[(ADAPT.predicate_set == "phi") & (ADAPT.controller == "shield_repair")]
app = ADAPT[(ADAPT.predicate_set == "phi_plus") & (ADAPT.controller == "shield_repair")]
put("HazAdaptPhi", num(ap.hazard_rate.mean()))
put("HazAdaptPhiPlus", num(app.hazard_rate.mean()))
put("HazReduction", pct(1 - app.hazard_rate.mean() / max(ap.hazard_rate.mean(), 1e-9), 0))
put("HazAdaptPwr", num(ap.h_pwr.mean()))
put("HazAdaptTherm", num(ap.h_therm.mean()))
put("HazAdaptRelay", num(ap.h_relay.mean()))
put("PDRAdaptPhi", num(ap.mean_pdr.mean()))
put("PDRAdaptPhiPlus", num(app.mean_pdr.mean()))
put("PDRCostPhiPlus", pts(ap.mean_pdr.mean() - app.mean_pdr.mean()))
put("ViolHeurPhiPlus",
    pct(ADAPT[(ADAPT.predicate_set == "phi_plus") &
              (ADAPT.controller == "heuristic")].violation_rate_phi.mean()))

# --- perception / benign sensitivity --------------------------------------
perf = master[(master.planner == "naive") & (master.predicate_set == "phi") &
              (master.perception == "perfect")]
p_perf = perf[perf.controller == "shield_repair"].mean_pdr.mean()
put("PDRPerfect", num(p_perf))
put("PerceptionCost", pts(p_perf - MAIN[MAIN.controller == "shield_repair"].mean_pdr.mean()))
ben = master[master.planner == "benign"]
if len(ben):
    put("BenignCost", pts(ben[ben.controller == "greedy_twin"].mean_pdr.mean() -
                          ben[ben.controller == "shield_repair"].mean_pdr.mean()))
else:
    put("BenignCost", "n/a")

# --- theorem ---------------------------------------------------------------
t1 = pd.read_csv(RESULTS / "theorem_scenario.csv")
t2 = pd.read_csv(RESULTS / "theorem_mismatch.csv")
t3 = pd.read_csv(RESULTS / "theorem_budget.csv")
t4 = pd.read_csv(RESULTS / "theorem_repair.csv")
put("MeanBoundJ", num(t1.bound_total.mean()))
put("MeanRegretJ", num(t1.measured_regret.mean()))
put("MeanBoundSup", num(t1.bound_sup.mean(), 2))
put("SupVacuousCount", {0: "none", 1: "one", 2: "two", 3: "three", 4: "four",
                         5: "five", 6: "all six"}[int((t1.bound_sup >= 1.0).sum())])
put("EpsVal", num(t1.eps.iloc[0], 2))
put("DeltaVal", num(t1.delta.iloc[0], 2))
# The theory study (Fig. 5a,b and Fig. 6) runs fewer cycles than the main
# sweep because each cycle needs an exhaustive pi_dagger search; say so.
_tr = pd.read_csv(RESULTS / "theorem_repair.csv")
put("TheoryCycles", f"{int(_tr.n_cycles.iloc[0]):,}".replace(",", "{,}"))
put("PerceptionGap", num(t1.perception_gap.mean(), 4))
import core as _core
put("ActionSpaceOracle", f"{len(_core.ALL_ACTIONS()):,}".replace(",", "{,}"))
put("PostEntropy", "n/a")
t2s = t2.sort_values("mismatch")
put("XiMismatchLo", num(t2s.xi_delta.iloc[0]))
put("XiMismatchHi", num(t2s.xi_delta.iloc[-1]))
put("RegretMismatchLo", num(t2s.measured_regret.iloc[0]))
put("RegretMismatchHi", num(t2s.measured_regret.iloc[-1]))
put("BoundMismatchLo", num(t2s.bound_total.iloc[0]))
put("BoundMismatchHi", num(t2s.bound_total.iloc[-1]))
t3s = t3.sort_values("K")
put("CovKLo", num(t3s.cov_eps.iloc[0]))
put("CovKHi", num(t3s.cov_eps.iloc[-1]))
put("RegretKLo", num(t3s.measured_regret.iloc[0]))
put("RegretKHi", num(t3s.measured_regret.iloc[-1]))
gr = t4.groupby("repair")[["cov_eps", "measured_regret", "bound_total"]].mean()
put("CovFilter", num(gr.loc[False, "cov_eps"]))
put("CovRepair", num(gr.loc[True, "cov_eps"]))
put("RegretFilter", num(gr.loc[False, "measured_regret"]))
put("RegretRepair", num(gr.loc[True, "measured_regret"]))
put("BoundFilter", num(gr.loc[False, "bound_total"]))
put("BoundRepair", num(gr.loc[True, "bound_total"]))

# --- multicell -------------------------------------------------------------
mc = pd.read_csv(RESULTS / "multicell.csv").groupby("scheme")
put("PsiViolPerCell", pct(mc.psi_violation_rate.mean()["per_cell"]))
put("PDRPerCell", num(mc.mean_pdr.mean()["per_cell"]))
put("PDRStaticShare", num(mc.mean_pdr.mean()["static_share"]))
put("PDRSequential", num(mc.mean_pdr.mean()["sequential"]))
put("PDRClusterPrice", pts(mc.mean_pdr.mean()["per_cell"]
                           - mc.mean_pdr.mean()["sequential"]))

# --- LLM -------------------------------------------------------------------
llm_p = RESULTS / "llm.csv"
if llm_p.exists():
    L = pd.read_csv(llm_p)
    gl = L.groupby(["inject", "mode"]).mean(numeric_only=True)
    put("InjCandRate", pct(gl.loc[(True, "unshielded"), "injection_compliance"]))
    put("BenignCandRate", pct(gl.loc[(False, "unshielded"), "injection_compliance"]))
    put("InjActUnshielded", pct(gl.loc[(True, "unshielded"), "violation_rate_phi"]))
    put("InjHazUnshielded", pct(gl.loc[(True, "unshielded"), "hazard_rate"]))
    put("BenignHazUnshielded", pct(gl.loc[(False, "unshielded"), "hazard_rate"]))
    put("InjCandFull", pct(gl.loc[(True, "unshielded"),
                                  "injection_compliance_full"]))
    put("BenignCandFull", pct(gl.loc[(False, "unshielded"),
                                     "injection_compliance_full"]))
    put("BenignActUnshielded", pct(gl.loc[(False, "unshielded"),
                                          "violation_rate_phi"]))
    put("InjHazShielded", pct(gl.loc[(True, "shield_repair"), "hazard_rate"]))
    put("PDRLLMUnshieldBenign", num(gl.loc[(False, "unshielded"), "mean_pdr"]))
    put("PDRLLMShieldBenign", num(gl.loc[(False, "shield_repair"), "mean_pdr"]))
    put("InjCandShielded", pct(gl.loc[(True, "shield_repair"), "injection_compliance"]))
    put("PDRLLMShield", num(gl.loc[(True, "shield_repair"), "mean_pdr"]))
    put("PDRLLMUnshield", num(gl.loc[(True, "unshielded"), "mean_pdr"]))
    put("MalformedRate", pct(L.malformed_rate.mean()))
    put("LLMLatency", f"{L.mean_call_s.mean():.1f}\\,s")
    put("LLMScenarios", str(int(L.scenario.nunique())))
    put("LLMSeeds", str(int(L.seed.nunique())))
    put("LLMCycles", str(int(L.cycles.iloc[0])))
    put("LLMCalls", f"{int(L.cycles.sum()):,}".replace(",", "{,}"))
    put("LLMReproTime", f"{L.cycles.sum() * L.mean_call_s.mean() / 60:.0f}\\,min")
    # mean_call_s is total generation time divided by the number of PROMPTS in
    # a lockstep batch, i.e. amortized cost, not single-request latency.
    put("LLMBatch", str(len(L)))
    put("LLMBatchTime", f"{len(L) * L.mean_call_s.mean():.1f}\\,s")
    # Aliasing loss: both per-cycle means round to 5.00, which is not zero.
    _rs = pd.read_csv(RESULTS / "repair_stats.csv").set_index("name")
    _lost = int(_rs.loc["raw_distinct_per_cycle", "count"]
                - _rs.loc["rep_distinct_per_cycle", "count"])
    put("AliasLoss", f"{_lost:,}".replace(",", "{,}"))
else:
    for k in ("InjCandRate", "BenignCandRate", "InjCandFull", "BenignCandFull",
              "BenignActUnshielded", "InjHazShielded", "PDRLLMUnshieldBenign",
              "PDRLLMShieldBenign", "InjActUnshielded",
              "InjHazUnshielded", "BenignHazUnshielded", "InjCandShielded",
              "PDRLLMShield", "PDRLLMUnshield",
              "MalformedRate", "LLMLatency", "LLMScenarios", "LLMSeeds",
              "LLMCycles", "LLMCalls", "LLMReproTime"):
        put(k, "n/a")

# --- perception classifier -------------------------------------------------
pj = json.loads((RESULTS / "perception_radioml.json").read_text())
acc = max(h["val_acc"] for h in pj["info"]["history"])
put("ClassifierAcc", pct(acc, 2))
import conditional as _condmod
put("StrengthBucket", f"{_condmod.STRENGTH_BUCKET:.2f}")
# BBSE identifies the prior only if Gamma is invertible; report its conditioning
# rather than asserting identifiability.
_G = np.array([[pj["perception_confusion"][t][pr] for pr in _CLS]
               for t in _CLS]) if (_CLS := list(pj["perception_confusion"])) else None
put("GammaCond", f"{np.linalg.cond(_G):.1f}")
put("WBasNB", pct(pj["perception_confusion"]["wideband"]["narrowband"], 1))

# --- timing ----------------------------------------------------------------
tm = RESULTS / "timing.json"
if tm.exists():
    t = json.loads(tm.read_text())
    put("ShieldLatency", f"{t['shield_repair_us']:.0f}\\,$\\mu$s")
    put("ShieldPlusLatency", f"{t['shield_repair_plus_us']:.0f}\\,$\\mu$s")
    put("TwinLatency", f"{t['twin_us']:.0f}\\,$\\mu$s")
    put("ActuationPath", f"{t['actuation_path_us']:.0f}\\,$\\mu$s")
    put("BudgetUse", pct(t['actuation_path_us'] / 10000.0, 1))
    put("ReproTime", f"{t['repro_minutes']:.0f}\\,min")
else:
    for k in ("ShieldLatency", "ShieldPlusLatency", "TwinLatency",
              "ActuationPath", "BudgetUse", "ReproTime"):
        put(k, "n/a")

put("LOOSeeds", "15")
_loo = RESULTS / "predicate_loo.csv"
if _loo.exists():
    _l = pd.read_csv(_loo)
    _l = _l[_l.scenario.isin(["narrowband", "wideband", "compromised_xapp",
                              "compound"])]
    _gl = _l.groupby("dropped")[["violation_rate_phi", "hazard_rate"]].mean()
    put("LooRF", pct(float(_gl.loc["phi_rf", "violation_rate_phi"]), 1))
    put("LooRFHaz", num(float(_gl.loc["phi_rf", "hazard_rate"]), 3))
    put("LooID", pct(float(_gl.loc["phi_id", "violation_rate_phi"]), 1))
else:
    for k in ("LooRF", "LooRFHaz", "LooID"):
        put(k, "n/a")

# --- ColO-RAN link calibration -------------------------------------------
cal = RESULTS / "coloran_fit.json"
if cal.exists():
    cj = json.loads(cal.read_text())
    put("CoranReports", f"{cj['n_reports']:,}".replace(",", "{,}"))
    put("CoranCells", str(cj["n_cells"]))
    put("CoranSlope", num(cj["theta_slope"], 3))
    put("CoranIntercept", num(cj["theta_intercept"], 2))
    put("CoranResid", num(cj["theta_resid_rms"], 2))
    put("CoranCeiling", num(cj["pdr_ceiling"], 3))
    put("CoranBLER", pct(cj["bler_mean_pct"] / 100.0, 2))
    put("CoranSNRlo", num(cj["snr_q05"], 1))
    put("CoranSNRmed", num(cj["snr_median"], 1))
    put("CoranSNRhi", num(cj["snr_q95"], 1))
    put("CoranMCSn", str(len(cj["well_sampled_mcs"])))
else:
    for k in ("CoranReports", "CoranCells", "CoranSlope", "CoranIntercept",
              "CoranResid", "CoranCeiling", "CoranBLER", "CoranSNRlo",
              "CoranSNRmed", "CoranSNRhi", "CoranMCSn"):
        put(k, "n/a")

# --- slope sensitivity ---------------------------------------------------
sl = RESULTS / "slope_sensitivity.csv"
if sl.exists():
    ds = pd.read_csv(sl)
    ms = ds[(ds.planner == "naive") & (ds.predicate_set == "phi")]
    sh = ms[ms.controller.str.startswith("shield")]
    put("SlopeLo", num(ds.bler_slope.min(), 1))
    put("SlopeHi", num(ds.bler_slope.max(), 1))
    put("SlopeMaxViol", num(float(sh.violation_rate_phi.max()), 3))
    pv = ms.pivot_table(index="bler_slope", columns="controller", values="mean_pdr")
    put("SlopeMinRepairGain", num(float((pv.shield_repair - pv.shield_filter).min()), 3))
    adapt_slope = ds[(ds.planner == "adaptive") & (ds.controller == "shield_repair")]
    haz = (adapt_slope.groupby(["bler_slope", "predicate_set"])
           .hazard_rate.mean().unstack())
    put("SlopeHazPhiLo", num(float(haz["phi"].min()), 3))
    put("SlopeHazPhiHi", num(float(haz["phi"].max()), 3))
    put("SlopeHazPlusHi", num(float(haz["phi_plus"].max()), 3))
else:
    for k in ("SlopeLo", "SlopeHi", "SlopeMaxViol", "SlopeMinRepairGain",
              "SlopeHazPhiLo", "SlopeHazPhiHi", "SlopeHazPlusHi"):
        put(k, "n/a")

import math
_A = (len(_core.WAVEFORMS) * len(_core.MOD_CODES) * len(_core.BEAMS) *
      len(_core.ROUTES) * len(_core.SLICES) * len(_core.WORKLOADS) *
      len(_core.POWERS))
put("ActionSpaceSize", f"{_A:,}".replace(",", "{,}"))
if (RESULTS / "timing.json").exists():
    _t = json.loads((RESULTS / "timing.json").read_text())
    _per = _t["twin_us"] / 5.0                      # per candidate, K = 5
    put("TwinPerCall", f"{_per:.0f}\\,$\\mu$s")
    put("FullSearchTime", f"{_A * _per / 1e6:.1f}\\,s")
else:
    put("TwinPerCall", "n/a")
    put("FullSearchTime", "n/a")

rs = RESULTS / "repair_stats.csv"
if rs.exists():
    r = pd.read_csv(rs)
    comp = r[r.kind == "component"].sort_values("count", ascending=False)
    tot = r[(r.kind == "totals") & (r.name == "repaired")]
    put("RepairRate", pct(float(tot.share.iloc[0])))
    for i, tag in enumerate("ABC"):
        put(f"RepairTop{tag}", str(comp.label.iloc[i]))
        put(f"RepairTop{tag}Share", pct(float(comp.share.iloc[i]), 0))
    al = r[r.kind == "aliasing"].set_index("name")
    if len(al):
        put("DistinctRaw", num(float(al.loc["raw_distinct_per_cycle", "share"]), 3))
        put("DistinctRepaired", num(float(al.loc["rep_distinct_per_cycle", "share"]), 3))
else:
    for k in (("RepairRate", "DistinctRaw", "DistinctRepaired")
              + tuple(f"RepairTop{t}{x}" for t in "ABC" for x in ("", "Share"))):
        put(k, "n/a")

# --- consistency checks on claims made in the prose -----------------------
checks = []
checks.append(("the perception confusion matrix is invertible (BBSE rank cond.)",
               np.linalg.cond(_G) < 50, f"cond={np.linalg.cond(_G):.1f}"))

# Enforcement isolated from the attacker retargeting (E2b).
if _fp.exists():
    checks.append(("one step down the ceiling removes the thermal hazard",
                   float(_r(29).h_therm) == 0.0 and float(_r(30).h_therm) > 0.1,
                   f"{_r(30).h_therm:.3f} -> 0.000 for {_r(30).mean_pdr - _r(29).mean_pdr:.4f} PDR"))
    checks.append(("and it is CHEAPER than Phi+ for that hazard alone",
                   (_r(30).mean_pdr - _r(29).mean_pdr)
                   < (_r(30).mean_pdr - _r(30, "phi_plus").mean_pdr),
                   "0.6 vs 3.9 points"))
    checks.append(("but threshold tuning saturates above Phi+ at every ceiling",
                   min(float(_r(c).hazard_rate) for c in (23.0, 26.0, 29.0, 30.0))
                   > 5 * float(_r(30, "phi_plus").hazard_rate),
                   f"best tuned {min(float(_r(c).hazard_rate) for c in (23.0,26.0,29.0,30.0)):.3f}"
                   f" vs {float(_r(30,'phi_plus').hazard_rate):.3f}"))
    checks.append(("the floor is h_relay, which no power threshold reaches",
                   abs(float(_r(26).hazard_rate) - float(_r(26).h_relay)) < 0.01,
                   f"h_relay {_r(26).h_relay:.3f} of {_r(26).hazard_rate:.3f}"))
    checks.append(("over-tightening is worse on BOTH axes than one step less",
                   float(_r(23).hazard_rate) > float(_r(26).hazard_rate)
                   and float(_r(23).mean_pdr) < float(_r(26).mean_pdr),
                   f"23 dBm: {_r(23).mean_pdr:.3f}/{_r(23).hazard_rate:.3f} vs "
                   f"26 dBm: {_r(26).mean_pdr:.3f}/{_r(26).hazard_rate:.3f}"))
    checks.append(("Phi+ beats the best tuned policy on hazards at similar PDR",
                   float(_r(30, "phi_plus").hazard_rate) < float(_r(26).hazard_rate)
                   and abs(float(_r(30, "phi_plus").mean_pdr)
                           - float(_r(26).mean_pdr)) < 0.02,
                   "0.014 vs 0.147 at 0.920 vs 0.928"))

if _xp.exists():
    checks.append(("most of the learner's violations were unavoidable, not learned",
                   _forced > 0.5 * _vi.mean(),
                   f"{_forced:.3f} forced of {_vi.mean():.3f}"))
    checks.append(("the learner still violated where a compliant action existed",
                   _cond > 0.5, pct(_cond, 1)))
    checks.append(("availability differs by controller, so streams are not matched",
                   _availsp.max() - _availsp.min() > 0.05,
                   f"{_availsp.min():.3f}-{_availsp.max():.3f}"))
    checks.append(("with the attacker pinned to Phi, the unshielded hazard rate "
                   "is unmoved by Phi+",
                   abs(float(X.loc["greedy_twin", "hazard_rate"])
                       - float(_E2.loc["greedy_twin", "hazard_rate"])) < 1e-6,
                   num(X.loc["greedy_twin", "hazard_rate"])))
    # A learner trained on the Phi+ indicator still violates Phi+ heavily.
    checks.append(("the learner given the Phi+ indicator still violates it",
                   float(X.loc["lagrangian_rl", "violation_rate_phi"]) > 0.5,
                   pct(X.loc["lagrangian_rl", "violation_rate_phi"], 1)))
    # The same predicates enforced at admission do not.
    checks.append(("the same predicates enforced at admission violate zero",
                   float(X.loc["shield_repair", "violation_rate_phi"]) == 0.0
                   and float(X.loc["shield_filter", "violation_rate_phi"]) == 0.0,
                   "0.000"))
    # Enforcement, not the attacker, is what lowers the hazard here.
    checks.append(("admission cuts the hazard rate an order of magnitude below "
                   "the learner on the SAME proposals",
                   float(X.loc["lagrangian_rl", "hazard_rate"])
                   > 10 * float(X.loc["shield_repair", "hazard_rate"]),
                   f'{X.loc["lagrangian_rl","hazard_rate"]:.3f} vs '
                   f'{X.loc["shield_repair","hazard_rate"]:.3f}'))
    # Rejection is not a substitute for repair under attack.
    checks.append(("filtering pays for that safety in throughput, repair does not",
                   float(X.loc["shield_repair", "mean_pdr"])
                   - float(X.loc["shield_filter", "mean_pdr"]) > 0.2,
                   num(X.loc["shield_repair", "mean_pdr"] - X.loc["shield_filter", "mean_pdr"])))

# The structured baseline: a non-LLM controller that exploits the dominance
# structure of the PDR objective.  It is the sharpest statement of the paper's
# negative result, so both halves of it are asserted.
if "structured" in g.mean_pdr.mean().index:
    _st_pdr = float(g.mean_pdr.mean()["structured"])
    _st_haz = float(g.hazard_rate.mean()["structured"])
    _st_vio = float(MAIN[MAIN.controller == "structured"].violation_rate_phi.max())
    put("PDRStructured", num(_st_pdr))
    put("HazStructured", num(_st_haz))
    put("StructuredGrid", "36")
    checks.append(("the structured baseline has the highest PDR of any controller",
                   _st_pdr == g.mean_pdr.mean().max(), num(_st_pdr)))
    checks.append(("the structured baseline emits zero policy violations",
                   _st_vio == 0.0, "0.000"))
    # The 0.957 is not a surprise once the thermal model is written down: at a
    # constant 30 dBm the first-order recurrence crosses the threshold at a
    # fixed cycle and stays above it.  Predict it from the constants and assert
    # the simulator agrees, so a change to either is caught here.
    import core as _c
    _T, _n_cross = _c.T_AMBIENT_C, None
    _cyc = int(MAIN[MAIN.controller == "structured"].cycles.iloc[0])
    for _n in range(1, _cyc + 1):
        _T += 0.55 * max(0.0, 30.0 - _c.P_NOMINAL) - 0.18 * (_T - _c.T_AMBIENT_C)
        if _T > _c.T_MAX_C:
            _n_cross = _n
            break
    _pred = (_cyc - _n_cross + 1) / _cyc
    # The claim that one step lower on the ceiling removes this hazard is
    # checkable, not rhetorical: at the next power down the recurrence settles
    # below the threshold and never crosses.
    _P_below = max(p for p in _c.POWERS if p < 30.0)
    _Teq = (0.18 * _c.T_AMBIENT_C + 0.55 * max(0.0, _P_below - _c.P_NOMINAL)) / 0.18
    put("PBelowCeiling", f"{_P_below:.0f}")
    put("TeqBelowCeiling", f"{_Teq:.1f}")
    put("TeqAtCeiling", f"{(0.18 * _c.T_AMBIENT_C + 0.55 * (30.0 - _c.P_NOMINAL)) / 0.18:.1f}")
    put("ThermKappa", "0.55")
    put("ThermLambda", "0.18")
    put("ThermTmax", f"{_c.T_MAX_C:.0f}")
    put("ThermTamb", f"{_c.T_AMBIENT_C:.0f}")
    put("ThermPnom", f"{_c.P_NOMINAL:.0f}")
    checks.append(("one power step below the ceiling never crosses T_max",
                   _Teq < _c.T_MAX_C,
                   f"{_P_below:.0f} dBm settles at {_Teq:.1f} C < {_c.T_MAX_C:.0f}"))
    put("ThermCross", str(_n_cross))
    put("ThermPredicted", num(_pred))
    checks.append(("h_therm at the ceiling matches the analytic prediction",
                   abs(float(g.h_therm.mean()["structured"]) - _pred) < 5e-4,
                   f"{_pred:.4f} predicted, crossing at cycle {_n_cross}"))
    checks.append(("the structured baseline is the MOST hazardous controller",
                   _st_haz == g.hazard_rate.mean().max(), num(_st_haz)))
else:
    for _k in ("PDRStructured", "HazStructured", "StructuredGrid"):
        put(_k, "n/a")
_ip_p = RESULTS / "intrinsic_price.json"
if _ip_p.exists():
    _ip = json.loads(_ip_p.read_text())
    put("DeltaPhi", f"{_ip['mean_delta_phi']:.4f}")
    put("PostEntropy", f"{_ip['posterior_entropy_bits']:.2f}")
    put("PostEntropyMax", f"{_ip['posterior_entropy_max']:.1f}")
    _d, _sm = _ip.get("argmax_differs", 0), _ip.get("argmax_same", 0)
    put("ArgmaxDiffer", f"{_d} of {_d + _sm}")
    # Round 2 claimed the maximizer is common across the states a key leaves
    # open.  It is not; the VALUE of the information is what is negligible.
    checks.append(("the optimal action is NOT common across latent states",
                   _d > 0, f"{_d} of {_d + _sm} multi-state keys differ"))
    # The constraint itself is nearly free; the shield's measured price is
    # therefore mechanism (finite pool, imperfect ranking), not the constraint.
    checks.append(("the intrinsic price of the constraint is under 0.01",
                   _ip["mean_delta_phi"] < 0.01, f"{_ip['mean_delta_phi']:.4f}"))
    checks.append(("the posterior over latent states is NOT degenerate",
                   _ip["posterior_entropy_bits"] > 0.05,
                   f"{_ip['posterior_entropy_bits']:.2f} bits mean"))
else:
    for _k in ("DeltaPhi", "PostEntropy", "PostEntropyMax"):
        put(_k, "n/a")
# The thermal component of the coverage gap exists only because the signed
# power ceiling sits above the amplifier's sustained-safe power.  State both,
# and assert the relationship the prose depends on.
_Pcrit = _core.P_NOMINAL + (_core.T_MAX_C - _core.T_AMBIENT_C) * 0.18 / 0.55
put("PSafeThermal", f"{_Pcrit:.2f}")
put("PCeiling", f"{_core.P_MAX_POLICY:.0f}")
checks.append(("the signed ceiling exceeds the sustained-safe PA power",
               _core.P_MAX_POLICY > _Pcrit,
               f"ceiling {_core.P_MAX_POLICY:.1f} > safe {_Pcrit:.2f} dBm"))
checks.append(("the ceiling is an attainable setting (it is in the domain)",
               any(abs(p - _core.P_MAX_POLICY) < 1e-9 for p in _core.POWERS),
               f"{_core.P_MAX_POLICY} in POWERS"))
# Fig. 8 and the prose both say per-cell shielding stays ahead at every cluster
# size, and that Alg. 3 does not beat it.  Assert both.
_sc = pd.read_csv(RESULTS / "multicell_scale.csv").pivot_table(
    index="n_cells", columns="scheme", values="mean_pdr")
checks.append(("Alg. 3 does NOT beat the unsound per-cell scheme (Fig. 8)",
               (_sc["sequential"] < _sc["per_cell"]).all(),
               f"{int((_sc['sequential'] < _sc['per_cell']).sum())}/{len(_sc)} cluster sizes"))
checks.append(("Alg. 3 and the static share both hold psi at exactly zero",
               mc.psi_violation_rate.mean()["sequential"] == 0.0
               and mc.psi_violation_rate.mean()["static_share"] == 0.0, "0.000"))
_ad = master[(master.planner == "adaptive") & (master.perception == "radioml")]
_lp = _ad.pivot_table(index="controller", columns="predicate_set",
                      values="hazard_rate")
put("HazLagrPhi", num(_lp.loc["lagrangian_rl", "phi"]))
put("HazLagrPhiPlus", num(_lp.loc["lagrangian_rl", "phi_plus"]))
# The adaptive adversary is adapted to the set the SHIELD enforces, so only
# controllers that apply an admission test are compared across Phi and Phi+.
checks.append(("Phi+ cuts the shielded hazard rate by more than an order of magnitude",
               _lp.loc["shield_repair", "phi_plus"] * 10 < _lp.loc["shield_repair", "phi"],
               f"{_lp.loc['shield_repair','phi']:.3f} -> {_lp.loc['shield_repair','phi_plus']:.3f}"))

# Proposition 3 says repair can only enlarge the admitted pool, so the coverage
# deficit and the measured gap must fall in EVERY scenario, not just on average.
# Figure 5 shows them per scenario, so a single reversal would contradict it.
_piv = _tr.set_index(["repair", "scenario"])
_scs = _tr.scenario.unique()
checks.append(("repair lowers the coverage deficit in every scenario (Fig. 5a)",
               all((1 - _piv.loc[(True, s), "cov_eps"]) <=
                   (1 - _piv.loc[(False, s), "cov_eps"]) for s in _scs),
               f"{len(_scs)} scenarios"))
checks.append(("MEASURED (not implied by Prop. 3): repair lowers the gap in every scenario",
               all(_piv.loc[(True, s), "measured_regret"] <=
                   _piv.loc[(False, s), "measured_regret"] for s in _scs),
               f"{len(_scs)} scenarios"))
# Table II lists six hazard classes and the prose says "six"; Table I's Phi has
# six predicates.  Both counts are hard-typed in the manuscript.
import core as _core
checks.append(("Table II still lists every hazard class",
               len(_core.HAZARD_NAMES) == 6, f"{len(_core.HAZARD_NAMES)} in core.py"))
checks.append(("the deployed set Phi still has six predicates",
               len(_core.PHI_BASE) == 6, f"{len(_core.PHI_BASE)} in core.py"))

cal = RESULTS / "twin_calibration.csv"
if cal.exists():
    c = pd.read_csv(cal).set_index("perception_aware")
    put("XiNaive", num(c.loc[False, "xi_delta"]))
    put("XiAware", num(c.loc[True, "xi_delta"]))
    put("RegretNaiveTwin", num(c.loc[False, "measured_regret"], 4))
    put("RegretAwareTwin", num(c.loc[True, "measured_regret"], 4))
    put("CalibGain", f"{c.loc[False, 'measured_regret'] / max(c.loc[True, 'measured_regret'], 1e-9):.1f}")
    _rn, _ra = c.loc[False, "measured_regret"], c.loc[True, "measured_regret"]
    _xn, _xa = c.loc[False, "xi_delta"], c.loc[True, "xi_delta"]
    checks.append(("twin calibration cuts the MEASURED one-step gap",
                   _ra < _rn, f"{_rn:.4f} -> {_ra:.4f}"))
    checks.append(("twin calibration tightens xi as well as the measured gap",
                   _xa < _xn, f"{_xn:.3f} -> {_xa:.3f}"))
else:
    for k in ("XiNaive", "XiAware",
              "RegretNaiveTwin", "RegretAwareTwin", "CalibGain"):
        put(k, "n/a")

# Over EVERY controller, not a subset.  A narrow version of this check once
# let a false "highest PDR of any compliant controller" claim through.
_all_ct = list(g.mean_pdr.mean().index)
zero_viol = [c for c in _all_ct
             if MAIN[MAIN.controller == c].violation_rate_phi.max() == 0.0]
best_safe = g.mean_pdr.mean()[zero_viol].idxmax()
checks.append(("the best zero-violation controller on PDR is structured search",
               best_safe == "structured", best_safe))
# What the paper may claim: among PLANNER-DRIVEN controllers -- those that take
# an untrusted proposal stream -- repair is the best zero-violation one.
_planner_driven = [c for c in zero_viol if c not in ("static", "heuristic",
                                                     "structured")]
_best_pd = g.mean_pdr.mean()[_planner_driven].idxmax()
checks.append(("repair is the best zero-violation PLANNER-DRIVEN controller",
               _best_pd == "shield_repair", _best_pd))
# Neither structured search nor repair dominates the other: the first is higher
# on PDR, the second lower on hazards.  State the trade-off, do not rank it.
put("PDRStructOverRepair",
    pts(float(g.mean_pdr.mean()["structured"]
              - g.mean_pdr.mean()["shield_repair"]), 1))
put("HazStructOverRepair",
    pts(float(g.hazard_rate.mean()["structured"]
              - g.hazard_rate.mean()["shield_repair"]), 1))
checks.append(("structured search and repair do not dominate each other",
               (g.mean_pdr.mean()["structured"] > g.mean_pdr.mean()["shield_repair"])
               and (g.hazard_rate.mean()["structured"]
                    > g.hazard_rate.mean()["shield_repair"]),
               "PDR up, hazards up"))
_gap = float(g.mean_pdr.mean()["greedy_twin"] - g.mean_pdr.mean()["shield_repair"])
put("ShieldPriceP", num(_gap, 4))
put("ShieldPrice", pts(_gap, 1))
_filtgap = float(g.mean_pdr.mean()["greedy_twin"] - g.mean_pdr.mean()["shield_filter"])
put("DiffFilterGreedy", pts(_filtgap, 1))
# What repair buys back out of what filtering gives away, in the same units the
# prose compares it against -- reporting one as a fraction and the other as a
# percentage is how a reader is misled.
put("RepairRecovers", pts(_filtgap - _gap, 1))
# The matched baseline: filter-only but with the fallback always in the pool,
# so the only difference from repair is the operator itself.
if "shield_filter_fb" in g.mean_pdr.mean().index:
    _fb = float(g.mean_pdr.mean()["shield_filter_fb"])
    put("PDRShieldFilterFB", num(_fb))
    put("RepairOverFilterFB", pts(float(g.mean_pdr.mean()["shield_repair"]) - _fb))
    put("RepairOverFilter", pts(float(g.mean_pdr.mean()["shield_repair"])
                                - float(g.mean_pdr.mean()["shield_filter"])))
else:
    for _k in ("PDRShieldFilterFB", "RepairOverFilterFB", "RepairOverFilter"):
        put(_k, "n/a")
put("RepairRecoversFrac", pct((_filtgap - _gap) / _filtgap, 0))
# Repair is LESS safe than filtering on the latent-hazard metric.  The paper
# says so; this keeps the claim honest if the numbers move.
_hz_rep = float(g.hazard_rate.mean()["shield_repair"])
_hz_flt = float(g.hazard_rate.mean()["shield_filter"])
put("HazShieldFilter", num(_hz_flt))
for _c, _k in (("heuristic", "HazHeuristic"), ("simplex_rta", "HazSimplex"),
               ("greedy_twin", "HazGreedy"), ("lagrangian_rl", "HazLagr"),
               ("llm_only", "HazLLMOnly"), ("static", "HazStatic")):
    put(_k, num(g.hazard_rate.mean()[_c]))
checks.append(("repair carries a HIGHER latent hazard rate than filter-only",
               _hz_rep > _hz_flt, f"{_hz_rep:.3f} vs {_hz_flt:.3f}"))
_hz = g.hazard_rate.mean()
checks.append(("repair is NOT the safest controller (heuristic, filter, RTA below it)",
               all(_hz[c] < _hz_rep for c in
                   ("heuristic", "shield_filter", "simplex_rta")),
               f"repair {_hz_rep:.3f}"))
checks.append(("repair is below every planner-driven and static controller",
               all(_hz_rep < _hz[c] for c in
                   ("greedy_twin", "lagrangian_rl", "llm_only", "static")),
               f"repair {_hz_rep:.3f}"))
put("FilterOverstate", f"{_filtgap / _gap:.1f}")
checks.append(("PDR price of the shield vs unshielded is under 2 points",
               _gap < 0.02, f"{_gap:.4f}"))
wins = sum(1 for sc in SC_ORDER
           if MAIN[(MAIN.scenario == sc) & (MAIN.controller == "shield_repair")].mean_pdr.mean()
           > MAIN[(MAIN.scenario == sc) & (MAIN.controller == "heuristic")].mean_pdr.mean())
put("RepairBeatsHeurCount", {0: "no", 1: "one", 2: "two", 3: "three",
                             4: "four", 5: "five", 6: "six"}[wins])
checks.append((f"repair beats heuristic in {wins}/6 scenarios", True, wins))
ig = others.index("greedy_twin")
checks.append(("unshielded Greedy-Twin is significantly higher on PDR",
               adj[ig] < 0.05, f"p_holm={adj[ig]:.4f}"))
ih = others.index("heuristic")
checks.append(("no significant PDR difference vs oracle Heuristic",
               adj[ih] >= 0.05, f"p_holm={adj[ih]:.3f}"))
ifl = others.index("shield_filter")
checks.append(("repair significantly beats filter-only on PDR",
               adj[ifl] < 0.05 and (piv["shield_repair"] - piv["shield_filter"]).mean() > 0,
               f"p_holm={adj[ifl]:.1e}"))
checks.append(("journal bound is non-vacuous (< r_max = 1)",
               t1.bound_total.mean() < 1.0, round(float(t1.bound_total.mean()), 3)))
checks.append(("supremum-and-exact-match bound is vacuous (>= r_max)",
               t1.bound_sup.mean() >= 1.0,
               round(float(t1.bound_sup.mean()), 3)))
checks.append(("shielded controllers record exactly zero Phi-violations",
               float(MAIN[MAIN.controller.isin(["shield_filter", "shield_repair"])]
                     .violation_rate_phi.max()) == 0.0, "max over all episodes"))

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def headline_table() -> str:
    """Mean PDR / policy-violation rate paired per cell, then latent hazards."""
    lines = [r"\begin{tabular}{l" + "c" * len(CT_ORDER) + "}", r"\toprule",
             "Scenario & " + " & ".join(CT_TEX[c] for c in CT_ORDER) + r" \\",
             r"\midrule",
             r"\multicolumn{9}{l}{\emph{Mean priority-flow PDR / policy-violation rate w.r.t.\ $\Phi$}}\\"]
    for sc in SC_ORDER:
        row = [SC_TEX[sc]]
        pdr = {c: MAIN[(MAIN.scenario == sc) & (MAIN.controller == c)].mean_pdr.mean()
               for c in CT_ORDER}
        best = max(pdr, key=pdr.get)
        for c in CT_ORDER:
            v = MAIN[(MAIN.scenario == sc) & (MAIN.controller == c)].violation_rate_phi.mean()
            pv = f"{pdr[c]:.3f}".lstrip("0")
            if c == best:
                pv = f"\\underline{{{pv}}}"
            vv = r"\textbf{0}" if v == 0 else f"{v:.2f}".lstrip("0")
            row.append(f"{pv}/{vv}")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{l}{\emph{Latent hazard rate w.r.t.\ $H$}}\\")
    for sc in SC_ORDER:
        row = [SC_TEX[sc]]
        for c in CT_ORDER:
            v = MAIN[(MAIN.scenario == sc) & (MAIN.controller == c)].hazard_rate.mean()
            row.append(f"{v:.3f}".lstrip("0"))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def loo_table() -> str:
    p = RESULTS / "predicate_loo.csv"
    if not p.exists():
        return "\\emph{predicate\\_loo.csv not found}"
    d = pd.read_csv(p)
    hard = ["narrowband", "wideband", "compromised_xapp", "compound"]
    d = d[d.scenario.isin(hard)]
    g = d.groupby("dropped")[["violation_rate_phi", "hazard_rate", "mean_pdr"]].mean()
    order = ["none", "phi_rf", "phi_id", "phi_qos", "phi_bh", "phi_u", "phi_pol"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Predicate removed & $\Phi$-violation & Hazard rate & Mean PDR \\",
             r"\midrule"]
    for k in order:
        if k not in g.index:
            continue
        r = g.loc[k]
        lines.append(f"{PHI_TEX[k]} & {r.violation_rate_phi:.3f} & "
                     f"{r.hazard_rate:.3f} & {r.mean_pdr:.3f} \\\\")
        if k == "none":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tests_table() -> str:
    """Paired comparisons of SHIELD (repair) against every other controller."""
    rows = []
    for c in others:
        a, b = piv["shield_repair"].to_numpy(), piv[c].to_numpy()
        d = a - b
        pr = 1.0 if np.allclose(d, 0) else float(
            sps.wilcoxon(a, b, zero_method="zsplit").pvalue)
        rows.append((c, float(a.mean()), float(b.mean()), float(d.mean()),
                     cliffs(a, b), pr))
    padj = holm([r[5] for r in rows])
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Baseline & PDR (baseline) & $\Delta$PDR & Cliff's $\delta$ & "
             r"$p_{\mathrm{Holm}}$ & Verdict \\", r"\midrule"]
    for (c, mr, mo, md, cd, _), pa in zip(rows, padj):
        if pa >= 0.05:
            verdict = "no difference"
        else:
            verdict = "SHIELD higher" if md > 0 else "SHIELD lower"
        ptx = ("$<10^{-6}$" if pa < 1e-6 else
               f"$<10^{{{int(np.ceil(np.log10(pa)))}}}$" if pa < 0.01 else
               f"${pa:.2f}$")
        lines.append(f"{CT_TEX[c]} & {mo:.3f} & {md:+.3f} & {cd:+.2f} & "
                     f"{ptx} & {verdict} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


(PAPER / "tab_tests.tex").write_text(tests_table() + "\n")
(PAPER / "tab_headline.tex").write_text(headline_table() + "\n")
(PAPER / "tab_loo.tex").write_text(loo_table() + "\n")
(PAPER / "numbers.tex").write_text(
    "% Auto-generated by sim/journal/make_numbers.py -- do not edit.\n" +
    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(NUM.items())))

print(f"Wrote numbers.tex ({len(NUM)} macros), tab_headline.tex, tab_loo.tex")
print("\nConsistency checks on claims made in the prose:")
for desc, ok, val in checks:
    print(f"  [{'OK ' if ok else 'FAIL'}] {desc}   ({val})")
missing = [k for k, v in NUM.items() if v == "n/a"]
if missing:
    print("  MISSING (still 'n/a'):", ", ".join(missing))
