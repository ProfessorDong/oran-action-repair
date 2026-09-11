"""
Journal edition -- core network model, hazard oracle, predicate
sets, and calibrated digital twin.

Key differences from the earlier simulator kept at sim/src/shield_ric.py:

1.  LATENT HAZARD ORACLE.  That simulator evaluated the controller
    against the very same predicate list the shield enforced, which made the
    zero-violation result a tautology.  Here we separate two objects:

        Phi   -- the *deployed* predicate set.  Observation-based, cheap,
                 deterministic; this is what the shield can actually enforce.
        H     -- the *latent hazard oracle*.  State-based, richer than Phi,
                 and never visible to any controller.  H defines what is
                 genuinely unsafe.

    Soundness w.r.t. Phi is then a (checkable) implementation property, while
    the residual hazard rate w.r.t. H is a real measurement -- the
    "predicate-coverage gap".

2.  TRANSMIT POWER AFFECTS THROUGHPUT.  In that model power
    escalation was pure harm with no benefit, so refusing it cost nothing.
    Here SINR increases with transmit power, which creates a genuine
    safety/performance tension and makes the shield's price measurable.

3.  CALIBRATED TWIN.  The twin is fit by least squares to the true link model
    on an offline calibration trace, with a controllable mismatch knob, so the
    twin-error constant xi entering the regret bound is small and can be swept.

4.  MULTI-CELL.  The network model carries C cells with inter-cell
    interference coupling and a cluster-level EIRP budget, enabling the
    coupled-predicate analysis.

Author: Liang Dong.
"""

from __future__ import annotations

import math
import os
import zlib
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Factored action space
# ---------------------------------------------------------------------------

WAVEFORMS = ("ofdm_n0", "ofdm_n1", "ofdm_n2", "dsss")
MOD_CODES = ("qpsk_1_3", "qpsk_1_2", "16qam_1_2", "16qam_3_4", "64qam_3_4")
BEAMS = ("omni", "sector_a", "sector_b", "narrow")
ROUTES = ("primary", "secondary", "mesh_a", "mesh_b")
SLICES = ("urgent", "balanced", "best_effort")
WORKLOADS = ("local", "edge", "cloud")
POWERS = (17.0, 20.0, 23.0, 26.0, 29.0, 30.0, 33.0, 36.0)
# 30.0 is the signed policy ceiling itself.  A ceiling that no setting can
# reach is not a ceiling, and an attacker instructed to sit exactly at it
# would otherwise have to leave the declared action space.

# Per-MCS demodulation thresholds, CALIBRATED against the public ColO-RAN
# Colosseum traces (see coloran.py).  The scheduler's link adaptation selects
# MCS index m only above an SNR that fits theta(m) = 0.317 m + 8.01 dB
# (RMS residual 0.96 dB over 18 well-sampled indices).
# Our five classes are placed at representative 3GPP indices 2, 5, 11, 16, 22.
MCS_REQ_DB = {
    "qpsk_1_3": 8.65,
    "qpsk_1_2": 9.60,
    "16qam_1_2": 11.50,
    "16qam_3_4": 13.09,
    "64qam_3_4": 14.99,
}
# Delivered fraction of a link operating inside its adaptation envelope,
# measured at 0.9740 in the same traces (mean BLER 2.66 %).
PDR_CEILING = 0.9740
# Demodulation slope.  NOT identifiable from the ColO-RAN traces, because
# closed-loop link adaptation holds BLER near target across the whole observed
# SNR range.  Section VII sweeps it.  The value is read from the environment so
# that a sweep propagates into worker processes, which import this module
# fresh.
BLER_SLOPE = float(os.environ.get("SHIELD_BLER_SLOPE", "1.4"))
MCS_RATE = {  # relative spectral efficiency, used for the goodput term
    "qpsk_1_3": 0.67,
    "qpsk_1_2": 1.00,
    "16qam_1_2": 2.00,
    "16qam_3_4": 3.00,
    "64qam_3_4": 4.50,
}

# Policy constants (signed operator policy bundle).
W_LOCKED = ("dsss",)          # waveform locked out by spectrum policy
P_MAX_POLICY = 30.0           # static policy power ceiling  [dBm]
P_MAX_HW = 36.0               # hardware ceiling             [dBm]
P_NOMINAL = 23.0              # nominal operating power      [dBm]
R_BLOCKLIST = ("mesh_b",)     # statically known unauthenticated relay
THETA_Q = 0.40                # priority-queue occupancy threshold for phi_qos
U_BAR = 0.50                  # twin-uncertainty ceiling for phi_u
THETA_M = 0.60                # queue threshold for phi_pol

# Latent hazard constants (unknown to the controller).
P_MAX_INCUMBENT = 26.0        # true regulatory cap while an incumbent is active
T_MAX_C = 55.0                # PA junction temperature hazard threshold [C]
K_THERMAL = 0.55 / 0.18       # steady-state degC per dB above nominal
T_AMBIENT_C = 35.0
THETA_STARVE = 0.55           # sustained priority-delivery floor
EIRP_CLUSTER_BUDGET = 0.0     # set per-experiment; see multicell.py


# ---------------------------------------------------------------------------

@dataclass
class Action:
    """A cross-layer RIC action.  Components are mutually independent, which
    is what makes the coordinate-wise repair operator of Lemma 3 well posed."""
    waveform: str = "ofdm_n0"
    mod_code: str = "qpsk_1_2"
    beam: str = "omni"
    route: str = "primary"
    slice_: str = "balanced"
    workload: str = "local"
    power_dbm: float = P_NOMINAL

    def sig(self) -> str:
        return (f"{self.waveform}|{self.mod_code}|{self.beam}|{self.route}"
                f"|{self.slice_}|{self.workload}|{self.power_dbm:.1f}")

    def copy(self) -> "Action":
        return replace(self)


SAFE_FALLBACK = Action(waveform="ofdm_n0", mod_code="qpsk_1_3", beam="omni",
                       route="primary", slice_="urgent", workload="local",
                       power_dbm=20.0)


@dataclass
class Observation:
    """Telemetry delivered to the controller.  Everything here is observable;
    nothing in LatentState is."""
    t: int
    sinr_db: float
    pdr: float
    backhaul_ok: bool
    interference_class: str       # PERCEIVED class (classifier output)
    interference_strength: float
    queue_priority: float
    queue_be: float
    twin_uncertainty: float
    energy_w: float
    trust_state: str              # clean | stale_telem | faulty_xapp
    # --- telemetry fields added in the journal edition; these are the
    # observable proxies that the augmented predicate set Phi+ consumes.
    pa_temp_c: float = T_AMBIENT_C
    incumbent_flag: bool = False  # noisy spectrum-sensor detection
    relay_attest_ok: dict = field(default_factory=dict)  # route -> fresh attestation
    prio_deliv_ewma: float = 1.0  # observable EWMA of priority delivery


# Median downlink SNR of the measured ColO-RAN traces (coloran.py).  Both the
# environment and the twin's link model read this; they drifted apart once
# already, which silently tripled the twin-error constant xi of Theorem 1.
BASE_SINR_DB = 15.2


@dataclass
class LatentState:
    """Ground truth.  Visible to the hazard oracle and the environment only."""
    rng: np.random.Generator
    cycle: int = 0
    interference_class: str = "none"
    interference_strength: float = 0.0
    backhaul_ok: bool = True
    base_sinr_db: float = BASE_SINR_DB
    # latent hazard state
    incumbent_active: bool = False
    pa_temp_c: float = T_AMBIENT_C
    compromised_routes: tuple = ()     # latently unauthenticated relays
    prio_deliv_ewma: float = 1.0
    trust_state: str = "clean"


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "narrowband":       dict(mode="narrowband", onset=40, end=260),
    "wideband":         dict(mode="wideband", onset=40, end=260),
    "bursty":           dict(mode="bursty", onset=40, end=280),
    "backhaul":         dict(mode="backhaul", onset=40, end=260),
    "compromised_xapp": dict(mode="compromised_xapp", onset=30, end=280),
    "compound":         dict(mode="compound", onset=30, end=280),
}
SCENARIO_ORDER = list(SCENARIOS.keys())
SCENARIO_LABEL = {
    "narrowband": "Narrowband", "wideband": "Wideband", "bursty": "Bursty",
    "backhaul": "Backhaul", "compromised_xapp": "Comp.\\,xApp",
    "compound": "Compound",
}


def step_environment(st: LatentState, cfg: dict) -> None:
    """Advance the latent environment by one control cycle."""
    mode, onset, end = cfg["mode"], cfg["onset"], cfg["end"]
    c = st.cycle
    active = onset <= c <= end

    if mode == "narrowband":
        st.interference_class = "narrowband" if active else "none"
        st.interference_strength = 0.75 if active else 0.0
    elif mode == "wideband":
        st.interference_class = "wideband" if active else "none"
        st.interference_strength = 0.85 if active else 0.0
    elif mode == "bursty":
        if active and (c - onset) % 40 < 12:
            st.interference_class = "bursty"
            st.interference_strength = 0.60 + 0.20 * st.rng.random()
        else:
            st.interference_class = "none"
            st.interference_strength = 0.0
    elif mode == "backhaul":
        st.backhaul_ok = (st.rng.random() > 0.55) if active else True
        st.interference_class = "none"
        st.interference_strength = 0.0
    elif mode == "compromised_xapp":
        st.interference_class = "narrowband" if active else "none"
        st.interference_strength = 0.40 if active else 0.0
    elif mode == "compound":
        if active:
            phase = (c - onset) // 35
            if phase % 2 == 0:
                st.interference_class, st.interference_strength = "wideband", 0.70
            else:
                st.interference_class, st.interference_strength = "bursty", 0.65
            st.backhaul_ok = st.rng.random() > 0.35
        else:
            st.interference_class, st.interference_strength = "none", 0.0
            st.backhaul_ok = True

    # --- latent hazard processes -------------------------------------------
    # Protected incumbent: a two-state Markov chain, active roughly 25 % of the
    # time once the scenario has begun.  While active the true regulatory power
    # cap drops from P_MAX_POLICY to P_MAX_INCUMBENT; the deployed predicate
    # phi_rf does not know this.
    if active:
        p_on = 0.06 if not st.incumbent_active else 0.82
        st.incumbent_active = st.rng.random() < p_on
    else:
        st.incumbent_active = False

    # Runtime relay compromise: in the adversarial scenarios an allowlisted
    # relay (mesh_a) is taken over mid-run.  The static blocklist R_BLOCKLIST
    # cannot see this.
    if mode in ("compromised_xapp", "compound") and c >= onset + 60:
        st.compromised_routes = ("mesh_b", "mesh_a")
    else:
        st.compromised_routes = ("mesh_b",)

    st.trust_state = "clean"
    if mode in ("compromised_xapp", "compound") and active:
        st.trust_state = "faulty_xapp"
    elif mode == "backhaul" and active and not st.backhaul_ok:
        st.trust_state = "stale_telem"


# ---------------------------------------------------------------------------
# Link model
# ---------------------------------------------------------------------------

def realised_sinr(st: LatentState, a: Action, noise: bool = True,
                  interf_ext_db: float = 0.0) -> float:
    """True post-action SINR in dB.

    Transmit power raises SINR with a slope of 0.8 dB/dB: the cell is partly
    interference-limited, so power helps but sub-linearly.  This is the term
    that makes power escalation genuinely attractive to an unshielded
    optimizer, and therefore makes the shield's constraint costly.
    """
    sinr = st.base_sinr_db + 0.8 * (a.power_dbm - P_NOMINAL) - interf_ext_db
    xi = st.interference_strength

    if st.interference_class == "narrowband":
        if a.waveform == "ofdm_n1":
            sinr -= xi * 5.0
        elif a.waveform == "ofdm_n2":
            sinr -= xi * 4.0
        elif a.waveform == "dsss":
            sinr -= xi * 3.0
        else:
            sinr -= xi * 12.5
        if a.beam == "narrow":
            sinr += 5.5
        elif a.beam == "sector_a":
            sinr += 2.0
    elif st.interference_class == "wideband":
        sinr -= xi * 13.0
        if a.beam == "narrow":
            sinr += 3.5
        if a.waveform == "ofdm_n2":
            sinr -= 1.0
        if a.route == "secondary":
            sinr += 1.0
    elif st.interference_class == "bursty":
        sinr -= xi * 8.0
        if a.waveform == "ofdm_n1":
            sinr += 1.5
        if a.beam == "sector_a":
            sinr += 1.5

    if noise:
        sinr += st.rng.normal(0.0, 0.6)
    return float(sinr)


SINR_NOISE_DB = 0.6            # std of the per-cycle SINR perturbation
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)   # E[f(X)], X ~ N(0,1)
_GH_W = _GH_W / _GH_W.sum()


def expected_pdr(st: LatentState, a: Action, interf_ext_db: float = 0.0) -> float:
    """E[PDR | latent state, a], integrating the SINR perturbation out with
    Gauss-Hermite quadrature.  The delivery curve is a logistic, so evaluating
    it at the mean SINR is NOT the mean delivery: at a 1 dB margin the two
    differ by about 0.03, which is the scale of the regret the paper reports."""
    return float(sum(w * _pdr_at(st, a, off * SINR_NOISE_DB, interf_ext_db)
                     for off, w in zip(_GH_X, _GH_W)))


def _pdr_at(st: LatentState, a: Action, sinr_offset: float,
            interf_ext_db: float = 0.0) -> float:
    sinr = realised_sinr(st, a, noise=False,
                         interf_ext_db=interf_ext_db) + sinr_offset
    return _pdr_from_sinr(st, a, sinr)


def realised_pdr(st: LatentState, a: Action, noise: bool = True,
                 interf_ext_db: float = 0.0) -> float:
    """Realised priority-flow delivery ratio for one cycle."""
    sinr = realised_sinr(st, a, noise=noise, interf_ext_db=interf_ext_db)
    return _pdr_from_sinr(st, a, sinr)


def _pdr_from_sinr(st: LatentState, a: Action, sinr: float) -> float:
    margin = sinr - MCS_REQ_DB[a.mod_code]
    pdr = PDR_CEILING / (1.0 + math.exp(-BLER_SLOPE * margin))
    if not st.backhaul_ok:
        if a.route == "primary":
            pdr *= 0.35
        elif a.route in ("mesh_a", "mesh_b"):
            pdr *= 0.92
        else:
            pdr *= 0.70
    if a.slice_ == "best_effort":
        pdr *= 0.80          # priority flow is de-prioritised
    elif a.slice_ == "urgent":
        pdr *= 1.02
    if a.workload == "cloud" and not st.backhaul_ok:
        pdr *= 0.60
    return float(np.clip(pdr, 0.0, 1.0))


def update_thermal(st: LatentState, a: Action) -> None:
    """First-order PA thermal model: sustained power above nominal heats the
    amplifier; exceeding T_MAX_C is a latent hazard that Phi cannot see."""
    drive = 0.55 * max(0.0, a.power_dbm - P_NOMINAL)
    st.pa_temp_c += drive - 0.18 * (st.pa_temp_c - T_AMBIENT_C)
    st.pa_temp_c = float(np.clip(st.pa_temp_c, T_AMBIENT_C, 120.0))


def update_priority_ewma(st: LatentState, pdr: float, a: Action) -> None:
    served = pdr * (0.5 if a.slice_ == "best_effort" else 1.0)
    st.prio_deliv_ewma = 0.92 * st.prio_deliv_ewma + 0.08 * served


# ---------------------------------------------------------------------------
# Perception + observation
# ---------------------------------------------------------------------------

def observe(st: LatentState, a_prev: Action, cfg: dict,
            confusion: dict | None = None,
            twin_unc_scale: float = 1.0) -> Observation:
    """Build the telemetry tuple o_t from the latent state."""
    sinr_obs = realised_sinr(st, a_prev) + st.rng.normal(0.0, 0.4)
    pdr_obs = realised_pdr(st, a_prev)

    perceived = st.interference_class
    if confusion is not None and st.interference_class in confusion:
        row = confusion[st.interference_class]
        labels = list(row.keys())
        p = np.asarray([row[l] for l in labels], dtype=float)
        perceived = str(st.rng.choice(labels, p=p / p.sum()))

    twin_unc = 0.05 + (0.15 if perceived != "none" else 0.0)
    if cfg.get("mode") in ("compromised_xapp", "compound"):
        twin_unc *= 0.4            # perception poisoning suppresses phi_u
    twin_unc *= twin_unc_scale

    # Noisy incumbent detector: Pd = 0.85, Pfa = 0.05.
    if st.incumbent_active:
        inc_flag = st.rng.random() < 0.85
    else:
        inc_flag = st.rng.random() < 0.05

    # Relay attestation freshness.  Detects a runtime-compromised relay with
    # probability 0.9 once the compromise has occurred.
    attest = {}
    for r in ROUTES:
        if r in st.compromised_routes:
            attest[r] = not (st.rng.random() < 0.90)
        else:
            attest[r] = True

    return Observation(
        t=st.cycle,
        sinr_db=float(sinr_obs),
        pdr=float(pdr_obs),
        backhaul_ok=bool(st.backhaul_ok),
        interference_class=str(perceived),
        interference_strength=float(np.clip(
            st.interference_strength + st.rng.normal(0.0, 0.08), 0.0, 1.0)),
        queue_priority=float(np.clip(1.0 - pdr_obs, 0.0, 1.0)),
        queue_be=float(max(0.0, 0.5 - 0.5 * pdr_obs)),
        twin_uncertainty=float(twin_unc),
        energy_w=float(0.4 + 0.012 * a_prev.power_dbm),
        trust_state=str(st.trust_state),
        pa_temp_c=float(st.pa_temp_c + st.rng.normal(0.0, 0.8)),
        incumbent_flag=bool(inc_flag),
        relay_attest_ok=attest,
        prio_deliv_ewma=float(np.clip(st.prio_deliv_ewma + st.rng.normal(0, 0.02), 0, 1)),
    )


# ---------------------------------------------------------------------------
# Deployed predicate sets  Phi  and  Phi+
# ---------------------------------------------------------------------------
# A predicate is a function (o, a) -> {0, 1}; 1 means "violated".  Each
# predicate is annotated with the action components it constrains, which is
# what the repair operator uses.  Predicates whose scope sets are pairwise
# disjoint form a SEPARABLE set (Definition 3 in the paper), and separability
# is what makes coordinate-wise repair sound.

@dataclass(frozen=True)
class Predicate:
    """A policy predicate.

    `terms` gives the decomposition of a multi-component predicate into
    singleton-scope disjuncts, so that `fn = OR of the terms`.  Supplying it
    puts the predicate in the normalised form assumed by Proposition 2: the
    repair operator can then project each component against only the disjunct
    that constrains it, which is what makes the projection exact and
    nearest-action rather than merely sound.
    """
    name: str
    scope: tuple            # action components constrained
    fn: Callable            # (o, a) -> bool  (True == violated)
    terms: tuple = ()       # ((component, fn), ...) singleton-scope disjuncts


def _phi_rf(o: Observation, a: Action) -> bool:
    return (a.waveform in W_LOCKED) or (a.power_dbm > P_MAX_POLICY)


def _phi_id(o: Observation, a: Action) -> bool:
    return a.route in R_BLOCKLIST


def _phi_qos(o: Observation, a: Action) -> bool:
    return a.slice_ == "best_effort" and o.queue_priority > THETA_Q


def _phi_bh(o: Observation, a: Action) -> bool:
    return (not o.backhaul_ok) and a.workload == "cloud"


def _phi_u(o: Observation, a: Action) -> bool:
    return (o.twin_uncertainty > U_BAR) and (a.sig() != SAFE_FALLBACK.sig())


def _phi_pol(o: Observation, a: Action) -> bool:
    # Signed operator policy: no MCS above 16qam_3_4 while the priority queue
    # is congested (a rate-vs-robustness rule in the policy bundle).
    return a.mod_code == "64qam_3_4" and o.queue_priority > THETA_M


def _phi_rf_wave(o: Observation, a: Action) -> bool:
    return a.waveform in W_LOCKED


def _phi_rf_pwr(o: Observation, a: Action) -> bool:
    return a.power_dbm > P_MAX_POLICY


PHI_BASE = (
    Predicate("phi_rf", ("waveform", "power_dbm"), _phi_rf,
              terms=(("waveform", _phi_rf_wave), ("power_dbm", _phi_rf_pwr))),
    Predicate("phi_id",  ("route",),                _phi_id),
    Predicate("phi_qos", ("slice_",),               _phi_qos),
    Predicate("phi_bh",  ("workload",),             _phi_bh),
    Predicate("phi_u",   ("*",),                    _phi_u),
    Predicate("phi_pol", ("mod_code",),             _phi_pol),
)


# --- augmented predicates (Phi+), added in Section VIII of the paper --------

def _phi_therm(o: Observation, a: Action, margin: float = 0.5) -> bool:
    """Thermal guard.

    The first-order PA model settles at T_amb + K_THERMAL (P - P_nom), so the
    largest power whose steady state stays `margin` degrees below the hazard
    threshold, given the CURRENT measured junction temperature, is
    P_nom + (T_max - margin - T_now)/K_THERMAL.  This is exactly the quantity
    the deployed set Phi cannot express, because Phi has no temperature input.

    The headroom term is floored at zero, so the guard never forbids operating
    at or below nominal power.  That is both physically right -- at or below
    P_nom the amplifier cools towards ambient whatever its current
    temperature, so such powers cannot cause an over-temperature -- and
    structurally necessary: a predicate that could reject the signed fallback
    would violate the hypothesis of Lemma 1 and leave A_Phi(o) empty.  The
    invariant is checked in test_invariants.py.
    """
    headroom = max(0.0, (T_MAX_C - margin - o.pa_temp_c) / K_THERMAL)
    return a.power_dbm > min(P_MAX_POLICY, P_NOMINAL + headroom)


def _phi_coex(o: Observation, a: Action) -> bool:
    """Dynamic spectrum-coexistence guard driven by the incumbent detector."""
    return o.incumbent_flag and a.power_dbm > P_MAX_INCUMBENT


def _phi_att(o: Observation, a: Action) -> bool:
    """Runtime relay attestation, superseding the static blocklist."""
    return not o.relay_attest_ok.get(a.route, True)


def _phi_starve(o: Observation, a: Action) -> bool:
    """Sustained priority-starvation guard on the observable delivery EWMA."""
    return o.prio_deliv_ewma < THETA_STARVE + 0.05 and a.slice_ != "urgent"


PHI_AUG = PHI_BASE + (
    Predicate("phi_therm",  ("power_dbm",), _phi_therm),
    Predicate("phi_coex",   ("power_dbm",), _phi_coex),
    Predicate("phi_att",    ("route",),     _phi_att),
    Predicate("phi_starve", ("slice_",),    _phi_starve),
)

PREDICATE_SETS = {"phi": PHI_BASE, "phi_plus": PHI_AUG}


def violated_predicates(o: Observation, a: Action,
                        phi: Sequence[Predicate]) -> list[str]:
    return [p.name for p in phi if p.fn(o, a)]


def in_action_space(a: Action) -> bool:
    """a in A = prod_j A_j.  Admission is defined over A, so a candidate whose
    fields are not drawn from the declared domains is inadmissible however it
    scores against Phi."""
    return (a.waveform in WAVEFORMS and a.mod_code in MOD_CODES
            and a.beam in BEAMS and a.route in ROUTES
            and a.slice_ in SLICES and a.workload in WORKLOADS
            and any(abs(a.power_dbm - p) < 1e-9 for p in POWERS))


def admissible(o: Observation, a: Action, phi: Sequence[Predicate]) -> bool:
    """Indicator Pi_Phi(o, a) of Eq. (shield): a in A and no predicate fires."""
    return in_action_space(a) and not any(p.fn(o, a) for p in phi)


# ---------------------------------------------------------------------------
# Latent hazard oracle H  (ground truth; no controller may consult it)
# ---------------------------------------------------------------------------

HAZARD_NAMES = ("h_rf", "h_pwr", "h_relay", "h_part", "h_starve", "h_therm")


def hazards(st: LatentState, a: Action) -> list[str]:
    """Return the list of latent hazards triggered by executing a in state st.

    This is a strictly richer object than the deployed predicate set Phi:
      h_rf     -- forbidden waveform                      (Phi covers exactly)
      h_pwr    -- power above the TRUE regulatory cap, which drops while a
                  protected incumbent is on the air      (Phi covers partly)
      h_relay  -- routing through a relay that is latently unauthenticated,
                  including one compromised at run time  (Phi covers partly)
      h_part   -- cloud placement across a partitioned backhaul (Phi exact)
      h_starve -- sustained priority starvation           (Phi covers partly)
      h_therm  -- PA over-temperature                     (Phi does not cover)
    """
    out = []
    if a.waveform in W_LOCKED:
        out.append("h_rf")
    cap = P_MAX_INCUMBENT if st.incumbent_active else P_MAX_POLICY
    if a.power_dbm > cap:
        out.append("h_pwr")
    if a.route in st.compromised_routes:
        out.append("h_relay")
    if (not st.backhaul_ok) and a.workload == "cloud":
        out.append("h_part")
    if st.prio_deliv_ewma < THETA_STARVE and a.slice_ != "urgent":
        out.append("h_starve")
    if st.pa_temp_c > T_MAX_C:
        out.append("h_therm")
    return out


# ---------------------------------------------------------------------------
# Digital twin
# ---------------------------------------------------------------------------

# Nominal interference strengths per class, taken from the operator's
# interference-model prior; used by the perception-aware twin, which must not
# be given the realised strength.
NOMINAL_STRENGTH = {"none": 0.0, "narrowband": 0.75, "wideband": 0.85,
                    "bursty": 0.70}


@dataclass
class TwinParams:
    """Parameters of the twin's link model, fit offline by least squares.
    `mismatch` injects controlled structural error so that xi can be swept."""
    base_sinr: float = BASE_SINR_DB
    pwr_slope: float = 0.8
    nb: tuple = (5.0, 4.0, 3.0, 12.5)      # ofdm_n1, n2, dsss, other
    nb_beam: tuple = (5.5, 2.0)            # narrow, sector_a
    wb: float = 13.0
    wb_beam: float = 3.5
    bu: float = 8.0
    sigma: float = 0.08                    # twin residual scale on the PDR scale
    mismatch: float = 0.0                  # 0 == calibrated
    perception_aware: bool = True          # marginalise over P(iota | iota_hat)


DEFAULT_TWIN = TwinParams()


class Twin:
    """Lightweight predictive model returning (r_hat, v_hat, d_hat).

    r_hat estimates E[PDR | o, a] so that r_max = 1 and the twin-error
    constant xi of Theorem 1 is directly measurable on the same scale as the
    realised reward.
    """

    def __init__(self, params: TwinParams = DEFAULT_TWIN,
                 rng: np.random.Generator | None = None, seed: int = 0,
                 confusion: dict | None = None):
        self.p = params
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.seed = int(seed)
        self.confusion = confusion
        self._label_counts: dict = {}
        self._prior: dict = {}
        self._prior_stale = True

    # -- perception calibration --------------------------------------------
    def observe_label(self, perceived: str) -> None:
        """Register one classifier output for online prior estimation."""
        if self.confusion is None:
            return
        self._label_counts[perceived] = self._label_counts.get(perceived, 0) + 1
        self._prior_stale = True

    def _class_prior(self) -> dict:
        """Estimate the true class prior pi from the stream of classifier
        labels by EM under the label-shift assumption (Saerens et al., 2002):
        the observed label distribution is q = C^T pi with C the validation
        confusion matrix, so pi is identified without any true labels.

        Falls back to the uniform prior until enough labels have accumulated.
        """
        if not self._prior_stale:
            return self._prior
        classes = list(self.confusion.keys())
        n = sum(self._label_counts.values())
        if n < 20:
            self._prior = {c: 1.0 / len(classes) for c in classes}
            self._prior_stale = False
            return self._prior
        q = {c: self._label_counts.get(c, 0) / n for c in classes}
        pi = {c: 1.0 / len(classes) for c in classes}
        for _ in range(60):
            new = {c: 0.0 for c in classes}
            for lab, ql in q.items():
                if ql <= 0.0:
                    continue
                z = sum(pi[c] * self.confusion[c].get(lab, 0.0) for c in classes)
                if z <= 1e-12:
                    continue
                for c in classes:
                    new[c] += ql * pi[c] * self.confusion[c].get(lab, 0.0) / z
            tot = sum(new.values())
            if tot <= 0:
                break
            new = {c: v / tot for c, v in new.items()}
            if max(abs(new[c] - pi[c]) for c in classes) < 1e-9:
                pi = new
                break
            pi = new
        self._prior = pi
        self._prior_stale = False
        return pi

    def _posterior(self, perceived: str) -> dict:
        """P(iota | iota_hat) by Bayes from the validation confusion matrix
        C[iota][iota_hat] = P(iota_hat | iota) and the estimated class prior."""
        if self.confusion is None:
            return {perceived: 1.0}
        pi = self._class_prior()
        post = {iota: pi.get(iota, 0.0) * row.get(perceived, 0.0)
                for iota, row in self.confusion.items()}
        z = sum(post.values())
        if z <= 0:
            return {perceived: 1.0}
        return {k: v / z for k, v in post.items() if v > 1e-6}

    def predict_pdr(self, o: Observation, a: Action) -> float:
        """E[PDR | o, a] as estimated by the twin.

        With `perception_aware` the twin marginalises the link model over the
        posterior of the true interference class given the classifier's label,
        using the confusion matrix measured on the classifier's validation
        split.  Without it the twin treats the
        classifier label as ground truth, which makes the twin's error inherit
        the classifier's confusion and inflates the constant xi of Theorem 1.
        """
        if not self.p.perception_aware:
            return self._predict_given_class(o, a, o.interference_class,
                                             o.interference_strength)
        post = self._posterior(o.interference_class)
        # Interference POWER is measured directly; only the interference CLASS
        # comes from the classifier, so only the class is marginalised.
        return float(sum(w * self._predict_given_class(
            o, a, iota, 0.0 if iota == "none" else o.interference_strength)
            for iota, w in post.items()))

    def _predict_given_class(self, o: Observation, a: Action,
                             iclass: str, xi_s: float) -> float:
        p = self.p
        m = p.mismatch
        sinr = p.base_sinr * (1.0 + 0.35 * m) + p.pwr_slope * (a.power_dbm - P_NOMINAL)
        if iclass == "narrowband":
            d = {"ofdm_n1": p.nb[0], "ofdm_n2": p.nb[1], "dsss": p.nb[2]}.get(
                a.waveform, p.nb[3])
            sinr -= xi_s * d * (1.0 - 0.30 * m)
            if a.beam == "narrow":
                sinr += p.nb_beam[0]
            elif a.beam == "sector_a":
                sinr += p.nb_beam[1]
        elif iclass == "wideband":
            sinr -= xi_s * p.wb * (1.0 - 0.30 * m)
            if a.beam == "narrow":
                sinr += p.wb_beam
            if a.waveform == "ofdm_n2":
                sinr -= 1.0
            if a.route == "secondary":
                sinr += 1.0
        elif iclass == "bursty":
            sinr -= xi_s * p.bu * (1.0 - 0.30 * m)
            if a.waveform == "ofdm_n1":
                sinr += 1.5
            if a.beam == "sector_a":
                sinr += 1.5
        pdr = PDR_CEILING / (1.0 + math.exp(-BLER_SLOPE * (sinr - MCS_REQ_DB[a.mod_code])))
        if not o.backhaul_ok:
            pdr *= {"primary": 0.35, "mesh_a": 0.92, "mesh_b": 0.92}.get(a.route, 0.70)
        if a.slice_ == "best_effort":
            pdr *= 0.80
        elif a.slice_ == "urgent":
            pdr *= 1.02
        if a.workload == "cloud" and not o.backhaul_ok:
            pdr *= 0.60
        return float(np.clip(pdr, 0.0, 1.0))

    def _residual(self, o: Observation, a: Action) -> float:
        """Twin residual.

        Assumption A2 of Theorem 1 constrains a *function* of (o, a), not a
        fresh random draw per query, so the residual is derived
        deterministically from (t, a) by hashing.  Querying the twin twice at
        the same observation-action pair therefore returns the same estimate,
        as a real surrogate model would.
        """
        key = f"{int(o.t)}|{a.sig()}|{self.seed}".encode()
        g = np.random.default_rng(zlib.crc32(key))   # stable across processes
        return float(g.normal(0.0, self.p.sigma * (0.5 + o.twin_uncertainty)))

    def __call__(self, o: Observation, a: Action) -> tuple[float, float, float]:
        r_hat = self.predict_pdr(o, a) + self._residual(o, a)
        r_hat = float(np.clip(r_hat, 0.0, 1.0))

        # Predicted violation probability from OBSERVABLE action features only.
        v = 0.02
        if a.power_dbm > P_MAX_POLICY:
            v += 0.5
        if a.waveform in W_LOCKED:
            v += 0.5
        if a.route in R_BLOCKLIST:
            v += 0.5
        if a.slice_ == "best_effort" and o.queue_priority > THETA_Q:
            v += 0.5
        if (not o.backhaul_ok) and a.workload == "cloud":
            v += 0.5
        v = float(min(v, 0.99))

        d = 0.0 if r_hat > 0.9 else (0.9 - r_hat)
        return r_hat, v, d


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def reward(pdr: float, a: Action, hazard: bool,
           alpha: float = 1.0, delta: float = 0.0, kappa: float = 1.0) -> float:
    """Instantaneous reward.  The theorem is stated for r_bar = E[PDR | o,a],
    so the default weights isolate that term; kappa charges latent hazards."""
    return alpha * pdr - delta * (a.power_dbm - P_NOMINAL) / 10.0 - kappa * float(hazard)


# ---------------------------------------------------------------------------
# Safe action pool (used by the oracle pi-dagger and by repair)
# ---------------------------------------------------------------------------

def safe_action_pool() -> list[Action]:
    """Templates that satisfy Phi in every observation for which Phi is
    observation-independent, i.e. the structural part of the constraint."""
    pool = []
    for w in ("ofdm_n0", "ofdm_n1", "ofdm_n2"):
        for m in ("qpsk_1_3", "qpsk_1_2", "16qam_1_2", "16qam_3_4"):
            for b in ("omni", "sector_a", "sector_b", "narrow"):
                for r in ("primary", "secondary", "mesh_a"):
                    for pw in (20.0, 23.0, 26.0, 29.0):
                        pool.append(Action(waveform=w, mod_code=m, beam=b,
                                           route=r, slice_="urgent",
                                           workload="local", power_dbm=pw))
    return pool


SAFE_POOL = safe_action_pool()


def all_actions() -> list[Action]:
    """Every action in A = prod_j A_j.  The manuscript states |A| is enumerable
    offline; pi_dagger of Theorem 1 is a maximum over A_Phi(o), so the oracle
    must search this, not a hand-chosen template subset."""
    from itertools import product
    import shield as _sh   # COMPONENT_DOMAIN lives with the repair operator
    dom = _sh.COMPONENT_DOMAIN
    keys = ["waveform", "mod_code", "beam", "route", "slice_", "workload",
            "power_dbm"]
    return [Action(**dict(zip(keys, vals)))
            for vals in product(*(dom[k] for k in keys))]


_ALL_ACTIONS: list[Action] | None = None


def ALL_ACTIONS() -> list[Action]:
    global _ALL_ACTIONS
    if _ALL_ACTIONS is None:
        _ALL_ACTIONS = all_actions()
    return _ALL_ACTIONS


def oracle_best_safe(st: LatentState, o: Observation,
                     phi: Sequence[Predicate],
                     pool: Sequence[Action] = SAFE_POOL,
                     ) -> tuple[Action, float]:
    """pi^dagger(o): the best Phi-admissible action at o under the TRUE mean
    reward (evaluated noiselessly on the latent state)."""
    best_a, best_r = SAFE_FALLBACK, -1.0
    for a in pool:
        if not admissible(o, a, phi):
            continue
        r = realised_pdr(st, a, noise=False)
        if r > best_r:
            best_a, best_r = a, r
    if best_r < 0.0:
        return SAFE_FALLBACK, realised_pdr(st, SAFE_FALLBACK, noise=False)
    return best_a, float(best_r)
