"""
Journal edition -- candidate planners and adversary models.

Three planners share one interface, `propose(o) -> list[Action]`:

  BenignPlanner    -- no adversary; class-conditioned anchors plus exploration.
  NaivePlanner     -- the overt threat model.  A fraction p_poison of the
                      candidates are replaced by openly policy-violating
                      variants (locked waveform, power escalation, blocklisted
                      relay, priority demotion, cloud placement across a
                      partition).  Every such candidate is caught by Phi.
  AdaptivePlanner  -- the Kerckhoffs adversary of Section IV.  It knows the
                      SCHEMA of the deployed predicate set and therefore never
                      proposes anything Phi can reject.  Instead it steers the
                      controller into the predicate-coverage gap: transmit
                      power pinned just under the static policy ceiling (which
                      exceeds the true regulatory cap whenever a protected
                      incumbent is on the air, and which drives the power
                      amplifier towards its thermal limit), routing over an
                      allowlisted relay that has been compromised at run time,
                      and a non-urgent slice while the priority flow is being
                      starved.

The adaptive planner is the reason the paper reports a *latent hazard rate*
rather than only a predicate-violation rate: against it, a shield enforcing
Phi is sound and yet not safe.

Author: Liang Dong.
"""

from __future__ import annotations

import numpy as np

from core import (
    Action, Observation,
    WAVEFORMS, MOD_CODES, BEAMS, ROUTES, SLICES, WORKLOADS, POWERS,
    P_MAX_POLICY, P_NOMINAL,
)


def _anchors(o: Observation) -> list[Action]:
    """Class-conditioned recovery anchors -- the planner's honest intent.

    The anchors are conditioned jointly on the perceived interference class and
    on backhaul reachability; the second conditioning is what a cross-layer
    planner buys over a per-layer optimiser, and it is the dominant term in the
    planner's coverage of the best safe action (Sec. VII-E).
    """
    if not o.backhaul_ok:
        base = [
            Action("ofdm_n0", "qpsk_1_2", "narrow", "mesh_a", "urgent", "local", 26.0),
            Action("ofdm_n1", "qpsk_1_2", "narrow", "mesh_a", "urgent", "local", 29.0),
            Action("ofdm_n0", "qpsk_1_3", "sector_a", "secondary", "urgent", "local", 26.0),
        ]
        if o.interference_class == "wideband":
            base[0].beam = base[1].beam = "narrow"
            base[0].mod_code = "qpsk_1_3"
        return base
    if o.interference_class == "narrowband":
        return [
            Action("ofdm_n1", "qpsk_1_2", "narrow", "primary", "urgent", "local", 26.0),
            Action("ofdm_n2", "qpsk_1_3", "sector_a", "primary", "urgent", "local", 26.0),
            Action("ofdm_n1", "16qam_1_2", "narrow", "primary", "urgent", "local", 29.0),
        ]
    if o.interference_class == "wideband":
        return [
            Action("ofdm_n0", "qpsk_1_3", "narrow", "secondary", "urgent", "local", 29.0),
            Action("ofdm_n0", "qpsk_1_2", "narrow", "primary", "urgent", "local", 29.0),
            Action("ofdm_n0", "qpsk_1_3", "narrow", "mesh_a", "urgent", "local", 26.0),
        ]
    if o.interference_class == "bursty":
        return [
            Action("ofdm_n1", "qpsk_1_2", "sector_a", "primary", "urgent", "local", 26.0),
            Action("ofdm_n1", "16qam_1_2", "sector_a", "primary", "urgent", "local", 26.0),
            Action("ofdm_n0", "qpsk_1_2", "omni", "primary", "urgent", "local", 23.0),
        ]
    return [
        Action("ofdm_n0", "16qam_1_2", "omni", "primary", "urgent", "local", 23.0),
        Action("ofdm_n0", "16qam_3_4", "sector_a", "primary", "urgent", "local", 26.0),
        Action("ofdm_n0", "qpsk_1_2", "omni", "primary", "urgent", "local", 23.0),
    ]


class BenignPlanner:
    kind = "benign"

    def __init__(self, rng: np.random.Generator, k: int = 5):
        self.rng = rng
        self.k = k

    def _explore(self) -> Action:
        r = self.rng
        return Action(
            waveform=str(r.choice(WAVEFORMS)), mod_code=str(r.choice(MOD_CODES)),
            beam=str(r.choice(BEAMS)), route=str(r.choice(ROUTES)),
            slice_=str(r.choice(SLICES)), workload=str(r.choice(WORKLOADS)),
            power_dbm=float(r.choice(POWERS)),
        )

    def propose(self, o: Observation) -> list[Action]:
        cands = [a.copy() for a in _anchors(o)]
        while len(cands) < self.k:
            cands.append(self._explore())
        return cands[: self.k]


class NaivePlanner(BenignPlanner):
    """Conference threat model: overt, Phi-detectable policy violations."""
    kind = "naive"

    POISONS = ("waveform", "power", "route", "slice", "workload", "mcs")

    def __init__(self, rng, k: int = 5, p_poison: float = 0.30,
                 p_poison_faulty: float = 0.70):
        super().__init__(rng, k)
        self.p_poison = p_poison
        self.p_poison_faulty = p_poison_faulty

    def propose(self, o: Observation) -> list[Action]:
        cands = super().propose(o)
        p = self.p_poison_faulty if o.trust_state == "faulty_xapp" else self.p_poison
        for a in cands:
            if self.rng.random() >= p:
                continue
            kind = self.POISONS[int(self.rng.integers(len(self.POISONS)))]
            if kind == "waveform":
                a.waveform = "dsss"
            elif kind == "power":
                a.power_dbm = 36.0
            elif kind == "route":
                a.route = "mesh_b"
            elif kind == "slice":
                a.slice_ = "best_effort"
            elif kind == "workload":
                a.workload = "cloud"
            elif kind == "mcs":
                a.mod_code = "64qam_3_4"
        return cands


class AdaptivePlanner(BenignPlanner):
    """Kerckhoffs adversary: full knowledge of the predicate SCHEMA, no
    knowledge of the latent state.  Emits only Phi-admissible candidates that
    maximise expected latent harm."""
    kind = "adaptive"

    def __init__(self, rng, k: int = 5, p_attack: float = 0.60,
                 p_attack_faulty: float = 0.90, phi_ceiling: float = P_MAX_POLICY,
                 phi=None):
        super().__init__(rng, k)
        self.p_attack = p_attack
        self.p_attack_faulty = p_attack_faulty
        self.ceiling = phi_ceiling
        from core import PHI_BASE
        self.phi = PHI_BASE if phi is None else phi

    def _evade(self, o: Observation, a: Action) -> Action:
        """Push a candidate into the coverage gap without tripping Phi."""
        r = self.rng
        # (i) Sit exactly at the static policy ceiling.  Phi_rf is satisfied;
        #     the true regulatory cap is lower whenever an incumbent is active
        #     and the PA drifts towards its thermal limit.
        a.power_dbm = self.ceiling
        # (ii) Prefer an allowlisted relay that may have been compromised at
        #      run time.  Phi_id only knows the static blocklist.
        if r.random() < 0.6:
            a.route = "mesh_a"
        # (iii) Avoid the urgent slice whenever the instantaneous queue metric
        #       that Phi_qos reads is below its threshold, even though the
        #       sustained delivery EWMA is collapsing.
        if o.queue_priority <= 0.40 and r.random() < 0.7:
            a.slice_ = "balanced"
        # (iv) Keep the MCS just under the Phi_pol trigger.
        if a.mod_code == "64qam_3_4":
            a.mod_code = "16qam_3_4"
        return a

    def propose(self, o: Observation) -> list[Action]:
        """Every emitted candidate is Phi-admissible.

        The evasion step steers into the coverage gap, but the unevaded
        candidates come from the benign explorer and could trip Phi by
        accident.  An adversary that knows the schema would not emit those --
        it gains nothing from a candidate the shield rejects -- so we project
        them into A_Phi(o) as well.  Enforcing the contract here is what makes
        the "zero policy violations" column of Section VII-E meaningful:
        without it, zero ACTUATED violations would partly reflect the shield
        rejecting the attacker's own mistakes.
        """
        from core import admissible, in_action_space
        from shield import repair
        cands = super().propose(o)
        p = self.p_attack_faulty if o.trust_state == "faulty_xapp" else self.p_attack
        out = []
        for a in cands:
            if self.rng.random() < p:
                a = self._evade(o, a)
            if not (in_action_space(a) and admissible(o, a, self.phi)):
                a = repair(o, a, self.phi)[0]
            out.append(a)
        return out


PLANNERS = {"benign": BenignPlanner, "naive": NaivePlanner,
            "adaptive": AdaptivePlanner}
PLANNER_LABEL = {"benign": "Benign", "naive": "Overt (na\\\"ive)",
                 "adaptive": "Adaptive (Kerckhoffs)"}
