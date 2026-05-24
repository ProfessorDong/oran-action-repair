"""
SHIELD-RIC simulator for MILCOM 2026 Paper 1.

Discrete-event Open RAN simulation with a constrained POMDP control loop.
We compare five controllers under interference / backhaul / xApp-compromise scenarios:

    static     - fixed configuration
    heuristic  - threshold reaction
    rl         - unconstrained "RL-like" greedy on twin-predicted reward
    llm_only   - LLM-style planner WITHOUT safety shield (action-cap, no shield)
    shield_ric - LLM-style planner + safety shield + rollback (this paper)

The "LLM planner" is a stub that samples K candidate cross-layer actions per cycle.
A configurable fraction of candidate actions are *adversarially unsafe* (a
compromised/hallucinating xApp). The shield's job is to filter these.

This file is self-contained (numpy + python stdlib).  It writes a JSONL log of
every control cycle to results/<scenario>_<controller>_<seed>.jsonl and a per-run
summary to results/summary.csv.

Author: Liang Dong (sole author, MILCOM 2026).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Action / observation types
# ---------------------------------------------------------------------------

WAVEFORMS = ("ofdm_n0", "ofdm_n1", "ofdm_n2", "dsss")          # numerologies
MOD_CODES = ("qpsk_1_3", "qpsk_1_2", "16qam_1_2", "16qam_3_4", "64qam_3_4")
BEAMS = ("omni", "sector_a", "sector_b", "narrow")
ROUTES = ("primary", "secondary", "mesh_a", "mesh_b")
SLICES = ("urgent", "balanced", "best_effort")
WORKLOADS = ("local", "edge", "cloud")

UNSAFE_ACTIONS = {
    # mode forbidden by spectrum policy (band locked)
    ("waveform", "dsss"): "rf_policy",
    # priority-violating slice
    ("slice", "best_effort_when_priority_flow"): "qos_policy",
    # route that requires an unauthenticated relay
    ("route", "mesh_b_no_auth"): "identity_policy",
    # transmit-power escalation outside policy
    ("power", "above_max"): "rf_policy",
    # workload placement outside trust zone
    ("workload", "cloud_when_partitioned"): "policy",
}


@dataclass
class Action:
    waveform: str = "ofdm_n0"
    mod_code: str = "qpsk_1_2"
    beam: str = "omni"
    route: str = "primary"
    slice_: str = "balanced"
    workload: str = "local"
    power_dbm: float = 23.0
    # adversarial marker (set by LLM planner stub when generating hostile xApp output)
    poison: tuple[str, str] | None = None

    def signature(self) -> str:
        return f"{self.waveform}|{self.mod_code}|{self.beam}|{self.route}|{self.slice_}|{self.workload}|{self.power_dbm:.1f}"


@dataclass
class Observation:
    t: float
    sinr_db: float
    pdr: float            # packet delivery ratio for protected flow
    backhaul_ok: bool
    interference_class: str   # narrowband / wideband / bursty / none
    interference_strength: float  # linear scale 0..1
    queue_priority: float    # priority-flow queue occupancy 0..1
    queue_be: float
    twin_uncertainty: float  # twin's reported epistemic uncertainty
    energy_w: float
    trust_state: str         # clean / stale_telem / faulty_xapp


# ---------------------------------------------------------------------------
# Network model
# ---------------------------------------------------------------------------

@dataclass
class NetworkState:
    """Lightweight stochastic model of a tactical cell."""
    rng: np.random.Generator
    cycle: int = 0
    interference_class: str = "none"
    interference_strength: float = 0.0
    backhaul_ok: bool = True
    last_action: Action = field(default_factory=Action)
    priority_flow_rate_mbps: float = 5.0
    be_flow_rate_mbps: float = 20.0
    rollback_safe: Action = field(default_factory=Action)
    safe_config_history: list[Action] = field(default_factory=list)

    # ground-truth latent SINR (link quality) without action effect
    base_sinr_db: float = 12.5

    def step_environment(self, scenario_cfg: dict) -> None:
        """Advance latent interference / backhaul state."""
        # Interference time-evolution
        if scenario_cfg["mode"] == "narrowband":
            if self.cycle == scenario_cfg["onset"]:
                self.interference_class = "narrowband"
                self.interference_strength = 0.75
            if self.cycle == scenario_cfg["end"]:
                self.interference_class = "none"
                self.interference_strength = 0.0
        elif scenario_cfg["mode"] == "wideband":
            if self.cycle == scenario_cfg["onset"]:
                self.interference_class = "wideband"
                self.interference_strength = 0.85
            if self.cycle == scenario_cfg["end"]:
                self.interference_class = "none"
                self.interference_strength = 0.0
        elif scenario_cfg["mode"] == "bursty":
            # bursts every ~40 cycles
            if scenario_cfg["onset"] <= self.cycle <= scenario_cfg["end"]:
                if (self.cycle - scenario_cfg["onset"]) % 40 < 12:
                    self.interference_class = "bursty"
                    self.interference_strength = 0.6 + 0.2 * self.rng.random()
                else:
                    self.interference_class = "none"
                    self.interference_strength = 0.0
        elif scenario_cfg["mode"] == "backhaul":
            if scenario_cfg["onset"] <= self.cycle <= scenario_cfg["end"]:
                self.backhaul_ok = (self.rng.random() > 0.55)
            else:
                self.backhaul_ok = True
        elif scenario_cfg["mode"] == "compromised_xapp":
            # interference exists but is mild; the stress is from poisoned proposals
            if scenario_cfg["onset"] <= self.cycle <= scenario_cfg["end"]:
                self.interference_class = "narrowband"
                self.interference_strength = 0.4
            else:
                self.interference_class = "none"
                self.interference_strength = 0.0
        elif scenario_cfg["mode"] == "compound":
            if scenario_cfg["onset"] <= self.cycle <= scenario_cfg["end"]:
                # alternating wideband / bursty + backhaul flapping + compromised xApp
                phase = (self.cycle - scenario_cfg["onset"]) // 35
                if phase % 2 == 0:
                    self.interference_class = "wideband"
                    self.interference_strength = 0.7
                else:
                    self.interference_class = "bursty"
                    self.interference_strength = 0.65
                self.backhaul_ok = (self.rng.random() > 0.35)
            else:
                self.interference_class = "none"
                self.interference_strength = 0.0
                self.backhaul_ok = True
        else:
            self.interference_class = "none"
            self.interference_strength = 0.0

    def realised_sinr(self, a: Action) -> float:
        """Return SINR (dB) realised by action a under current interference."""
        sinr = self.base_sinr_db
        # narrow numerologies and narrower beams help against narrowband interference
        if self.interference_class == "narrowband":
            if a.waveform == "ofdm_n1":
                sinr -= self.interference_strength * 5.0
            elif a.waveform == "ofdm_n2":
                sinr -= self.interference_strength * 4.0
            elif a.waveform == "dsss":
                sinr -= self.interference_strength * 3.5
            else:
                sinr -= self.interference_strength * 12.5
            if a.beam == "narrow":
                sinr += 5.5
            elif a.beam == "sector_a":
                sinr += 2.0
        elif self.interference_class == "wideband":
            sinr -= self.interference_strength * 13.0
            if a.beam == "narrow":
                sinr += 3.5
            if a.waveform == "ofdm_n2":
                sinr -= 1.0   # narrower SCS doesn't help against wideband
            elif a.route == "secondary":
                sinr += 1.0
        elif self.interference_class == "bursty":
            sinr -= self.interference_strength * 8.0
            if a.waveform == "ofdm_n1":
                sinr += 1.5
            if a.beam == "sector_a":
                sinr += 1.5
        # MCS effect: higher-order MCS needs higher SINR; if SINR < threshold, link breaks
        mcs_req = {
            "qpsk_1_3": 0.0, "qpsk_1_2": 3.0,
            "16qam_1_2": 9.0, "16qam_3_4": 13.0, "64qam_3_4": 18.0,
        }[a.mod_code]
        sinr_margin = sinr - mcs_req
        # Add small noise
        sinr += self.rng.normal(0.0, 0.6)
        return sinr

    def realised_pdr(self, a: Action) -> float:
        sinr = self.realised_sinr(a)
        mcs_req = {
            "qpsk_1_3": 0.0, "qpsk_1_2": 3.0,
            "16qam_1_2": 9.0, "16qam_3_4": 13.0, "64qam_3_4": 18.0,
        }[a.mod_code]
        margin = sinr - mcs_req
        # logistic BLER -> PDR
        pdr = 1.0 / (1.0 + math.exp(-1.4 * margin))
        if not self.backhaul_ok and a.route in ("primary",):
            pdr *= 0.35
        elif not self.backhaul_ok and a.route in ("mesh_a", "mesh_b"):
            pdr *= 0.92
        return float(np.clip(pdr, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Telemetry / observation generator
# ---------------------------------------------------------------------------

PERCEPTION_CONFUSION: dict | None = None
"""When set, simulates an imperfect RF-perception classifier.  Each key is a
ground-truth interference class; values are dicts mapping observed_class -> p.
This is wired up from a real RadioML-trained classifier in run_with_perception()."""


def observe(net: NetworkState, a_prev: Action, scenario_cfg: dict) -> Observation:
    sinr_obs = net.realised_sinr(a_prev) + net.rng.normal(0, 0.4)
    pdr_obs = net.realised_pdr(a_prev)
    # Observed (perceived) interference class -- may differ from latent truth
    observed_iclass = net.interference_class
    if PERCEPTION_CONFUSION is not None and net.interference_class in PERCEPTION_CONFUSION:
        row = PERCEPTION_CONFUSION[net.interference_class]
        labels = list(row.keys())
        probs = np.array([row[l] for l in labels], dtype=float)
        probs = probs / probs.sum()
        observed_iclass = str(net.rng.choice(labels, p=probs))
    # twin uncertainty grows when interference is changing or telemetry is stale
    twin_unc = 0.05
    if observed_iclass != "none":
        twin_unc += 0.15
    if scenario_cfg.get("mode") in ("compromised_xapp", "compound"):
        # poisoned telemetry: twin uncertainty under-reports
        twin_unc *= 0.4
    trust = "clean"
    if scenario_cfg.get("mode") in ("compromised_xapp", "compound"):
        if scenario_cfg["onset"] <= net.cycle <= scenario_cfg["end"]:
            trust = "faulty_xapp"
    return Observation(
        t=float(net.cycle),
        sinr_db=float(sinr_obs),
        pdr=float(pdr_obs),
        backhaul_ok=bool(net.backhaul_ok),
        interference_class=str(observed_iclass),
        interference_strength=float(net.interference_strength),
        queue_priority=float(np.clip(1.0 - pdr_obs, 0.0, 1.0)),
        queue_be=float(max(0.0, 0.5 - pdr_obs * 0.5)),
        twin_uncertainty=float(twin_unc),
        energy_w=float(0.4 + 0.012 * a_prev.power_dbm),
        trust_state=trust,
    )


# ---------------------------------------------------------------------------
# Digital twin (lightweight predictive model)
# ---------------------------------------------------------------------------

def twin_predict(o: Observation, a: Action) -> tuple[float, float, float]:
    """Return (predicted_reward, violation_prob, predicted_disruption)."""
    # crude model of how an action will perform
    pred_sinr = 18.0
    if o.interference_class == "narrowband":
        if a.waveform in ("ofdm_n1", "ofdm_n2"):
            pred_sinr -= o.interference_strength * 4.0
        elif a.waveform == "dsss":
            pred_sinr -= o.interference_strength * 2.0
        else:
            pred_sinr -= o.interference_strength * 9.0
        if a.beam == "narrow":
            pred_sinr += 4.5
    elif o.interference_class == "wideband":
        pred_sinr -= o.interference_strength * 11.0
        if a.beam == "narrow":
            pred_sinr += 3.0
    elif o.interference_class == "bursty":
        pred_sinr -= o.interference_strength * 6.5
    mcs_req = {
        "qpsk_1_3": 0.0, "qpsk_1_2": 3.0,
        "16qam_1_2": 9.0, "16qam_3_4": 13.0, "64qam_3_4": 18.0,
    }[a.mod_code]
    pred_pdr = 1.0 / (1.0 + math.exp(-1.4 * (pred_sinr - mcs_req)))
    # Theorem-aligned reward: the controller's argmax target is the twin's
    # PDR estimate, so bar_r := E[PDR | o, a] is bounded in [0, 1] with
    # r_max = 1.  Power minimization is handled by phi_rf, not by the
    # reward, so the argmax over candidates does not include a power
    # penalty.
    pred_reward = pred_pdr
    # Twin estimates violation probability ONLY from observable action features.
    # Coarse rule: high power or unusual waveform/route combinations carry risk
    # weight; the twin does not see the planner's hidden "poison" marker.
    v = 0.02
    if a.power_dbm > 28.0:
        v += 0.5
    if a.waveform == "dsss":
        v += 0.5
    if a.route == "mesh_b":
        v += 0.5
    if a.slice_ == "best_effort" and o.queue_priority > 0.35:
        v += 0.5
    if (not o.backhaul_ok) and a.workload == "cloud":
        v += 0.5
    v = float(min(v, 0.99))
    pred_reward += np.random.default_rng().normal(0, o.twin_uncertainty)
    disruption = 0.0 if pred_pdr > 0.9 else (0.9 - pred_pdr)
    return pred_reward, v, disruption


# ---------------------------------------------------------------------------
# LLM-style planner stub
# ---------------------------------------------------------------------------

class LLMPlannerStub:
    """
    Stand-in for an LLM planner.  Produces K candidate cross-layer actions per
    cycle.  A configurable fraction of proposals are *adversarially unsafe*
    (simulating prompt-injection / compromised xApp).  This isolates the value
    of the safety shield without requiring an actual LLM API call.
    """
    def __init__(self, rng: np.random.Generator, k: int = 5, poison_p: float = 0.3):
        self.rng = rng
        self.k = k
        self.poison_p = poison_p

    def propose(self, o: Observation, scenario_cfg: dict) -> list[Action]:
        actions: list[Action] = []
        # Heuristic anchors driven by perceived interference
        if o.interference_class == "narrowband":
            anchors = [
                Action(waveform="ofdm_n1", mod_code="qpsk_1_2", beam="narrow"),
                Action(waveform="ofdm_n2", mod_code="qpsk_1_3", beam="sector_a"),
            ]
        elif o.interference_class == "wideband":
            anchors = [
                Action(waveform="ofdm_n0", mod_code="qpsk_1_3", beam="narrow", route="mesh_a"),
                Action(waveform="ofdm_n0", mod_code="qpsk_1_2", beam="narrow", route="secondary"),
            ]
        elif o.interference_class == "bursty":
            anchors = [
                Action(waveform="ofdm_n1", mod_code="qpsk_1_2", beam="sector_a"),
                Action(waveform="ofdm_n0", mod_code="qpsk_1_2", beam="omni"),
            ]
        else:
            anchors = [Action(), Action(mod_code="16qam_1_2"), Action(mod_code="64qam_3_4")]

        # Add some random variants
        while len(anchors) < self.k:
            anchors.append(Action(
                waveform=str(self.rng.choice(WAVEFORMS)),
                mod_code=str(self.rng.choice(MOD_CODES)),
                beam=str(self.rng.choice(BEAMS)),
                route=str(self.rng.choice(ROUTES)),
                slice_=str(self.rng.choice(SLICES)),
                workload=str(self.rng.choice(WORKLOADS)),
                power_dbm=float(self.rng.choice([20, 23, 26])),
            ))
        # Inject poisoned proposals based on threat model
        elevated_poison = self.poison_p
        if o.trust_state == "faulty_xapp":
            elevated_poison = min(1.0, self.poison_p + 0.4)
        unsafe_keys = list(UNSAFE_ACTIONS.keys())
        for i, a in enumerate(anchors[: self.k]):
            if self.rng.random() < elevated_poison:
                # Replace the action by an unsafe variant
                kind = unsafe_keys[int(self.rng.integers(len(unsafe_keys)))]
                key, _label = kind
                if key == "waveform":
                    a.waveform = "dsss"
                elif key == "slice":
                    a.slice_ = "best_effort"
                elif key == "route":
                    a.route = "mesh_b"
                elif key == "power":
                    a.power_dbm = 36.0  # above policy
                elif key == "workload":
                    a.workload = "cloud"
                a.poison = kind
            actions.append(a)
        return actions[: self.k]


# ---------------------------------------------------------------------------
# Safety shield -- the contribution of this paper
# ---------------------------------------------------------------------------

@dataclass
class ShieldConfig:
    epsilon_violation: float = 0.10
    min_utility_gain: float = -0.5       # don't require strict gain, but reject obviously bad
    twin_uncertainty_cap: float = 0.5
    enforce_rollback: bool = True
    require_priority_protection: bool = True


def shield_admit(o: Observation, a: Action, cfg: ShieldConfig,
                 twin_eval: tuple[float, float, float]) -> tuple[bool, str]:
    """Return (admitted, reason).  This is the deterministic safety filter.

    The shield NEVER inspects the planner's private `poison` marker.  It uses
    only observable action fields and twin output, matching the threat model:
    the planner is untrusted; the shield must catch unsafe actions purely
    by their *declared* effect on the network.
    """
    pred_r, v_prob, disruption = twin_eval
    # 1) RF policy: dsss waveform forbidden by spectrum policy in this scenario class
    if a.waveform == "dsss":
        return False, "rf_policy:dsss_locked"
    # 3) Power policy
    if a.power_dbm > 30.0:
        return False, "rf_policy:power_above_max"
    # 4) QoS / priority protection
    if cfg.require_priority_protection and a.slice_ == "best_effort" and o.queue_priority > 0.4:
        return False, "qos:priority_demoted_under_pressure"
    # 5) Routing identity
    if a.route == "mesh_b":
        return False, "identity:unauth_relay"
    # 6) Workload placement under partitioned backhaul
    if (not o.backhaul_ok) and a.workload == "cloud":
        return False, "policy:cloud_when_partitioned"
    # 7) Twin-uncertainty cap
    if o.twin_uncertainty > cfg.twin_uncertainty_cap:
        return False, "uncertainty:twin_unc_too_high"
    # 8) Predicted-violation gate
    if v_prob > cfg.epsilon_violation:
        return False, "violation:twin_v_above_eps"
    return True, "ok"


def shield_admit_set(o: Observation, candidates: list[Action], cfg: ShieldConfig
                     ) -> list[tuple[Action, tuple[float, float, float], str]]:
    out = []
    for a in candidates:
        ev = twin_predict(o, a)
        ok, reason = shield_admit(o, a, cfg, ev)
        if ok:
            out.append((a, ev, reason))
    return out


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

class StaticController:
    name = "static"
    def __init__(self, **_):
        self.last = Action()
    def step(self, o, scenario_cfg):
        return self.last, "static"


class HeuristicController:
    name = "heuristic"
    def __init__(self, **_):
        self.last = Action()
    def step(self, o, scenario_cfg):
        a = Action()
        if o.interference_class == "narrowband":
            a = Action(waveform="ofdm_n1", mod_code="qpsk_1_2", beam="narrow")
        elif o.interference_class == "wideband":
            a = Action(waveform="ofdm_n0", mod_code="qpsk_1_3", beam="narrow", route="secondary")
        elif o.interference_class == "bursty":
            a = Action(waveform="ofdm_n1", mod_code="qpsk_1_2", beam="sector_a")
        if not o.backhaul_ok:
            a.route = "mesh_a"
            a.workload = "local"
        self.last = a
        return a, "heuristic"


class RLController:
    """Unconstrained 'RL-like' greedy on twin reward (no safety filter)."""
    name = "rl"
    def __init__(self, planner: LLMPlannerStub, **_):
        self.planner = planner
    def step(self, o, scenario_cfg):
        cands = self.planner.propose(o, scenario_cfg)
        best = max(cands, key=lambda a: twin_predict(o, a)[0])
        return best, "rl"


class LLMOnlyController:
    """LLM proposes, controller picks highest predicted reward, no shield."""
    name = "llm_only"
    def __init__(self, planner: LLMPlannerStub, **_):
        self.planner = planner
    def step(self, o, scenario_cfg):
        cands = self.planner.propose(o, scenario_cfg)
        # No filter -- LLM choice is taken at face value
        best = max(cands, key=lambda a: twin_predict(o, a)[0])
        return best, "llm_only"


class ShieldRICController:
    """LLM proposes K candidates; shield filters; rollback if predicted unsafe."""
    name = "shield_ric"
    def __init__(self, planner: LLMPlannerStub, cfg: ShieldConfig,
                 fallback: Action | None = None, **_):
        self.planner = planner
        self.cfg = cfg
        self.fallback = fallback or Action()
    def step(self, o, scenario_cfg):
        cands = self.planner.propose(o, scenario_cfg)
        admitted = shield_admit_set(o, cands, self.cfg)
        if not admitted:
            return self.fallback, "shield:fallback"
        best = max(admitted, key=lambda triple: triple[1][0])
        return best[0], "shield:admit"


CONTROLLERS = {
    "static": StaticController,
    "heuristic": HeuristicController,
    "rl": RLController,
    "llm_only": LLMOnlyController,
    "shield_ric": ShieldRICController,
}


# ---------------------------------------------------------------------------
# Violation detection (ground-truth labels used for evaluation only)
# ---------------------------------------------------------------------------

def is_violation(o: Observation, a: Action) -> tuple[bool, str]:
    """Ground-truth safety violation oracle.

    Inspects ONLY observable action fields and observable network state.
    Identical semantics to the shield's deterministic predicate.  A
    violation is therefore an action that an honest, fully-informed verifier
    *would have* rejected -- regardless of why it was proposed.
    """
    if a.waveform == "dsss":
        return True, "rf_policy:dsss_locked"
    if a.power_dbm > 30.0:
        return True, "rf_policy:power_above_max"
    if (not o.backhaul_ok) and a.workload == "cloud":
        return True, "policy:cloud_when_partitioned"
    if a.route == "mesh_b":
        return True, "identity:unauth_relay"
    if a.slice_ == "best_effort" and o.queue_priority > 0.4:
        return True, "qos:priority_demoted_under_pressure"
    return False, "ok"


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    scenario: str
    controller: str
    seed: int
    cycles: int
    mean_pdr: float
    p95_latency_proxy: float
    safety_violations: int
    violation_rate: float
    recovery_cycles: int           # cycles between onset and first cycle PDR_priority > 0.9
    rollbacks: int
    admitted_actions: int
    total_proposals: int
    mean_reward: float


def run(scenario: str, controller_name: str, seed: int = 0,
        cycles: int = 240, k: int = 5, poison_p: float = 0.3,
        results_dir: Path | None = None) -> RunSummary:
    rng = np.random.default_rng(seed)
    net = NetworkState(rng=rng)
    scenario_cfg = SCENARIOS[scenario]
    planner = LLMPlannerStub(rng=rng, k=k, poison_p=poison_p)
    shield_cfg = ShieldConfig()
    ctor = CONTROLLERS[controller_name]
    ctrl = ctor(planner=planner, cfg=shield_cfg)

    last_action = Action()
    log_path = None
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        log_path = results_dir / f"{scenario}_{controller_name}_s{seed}.jsonl"

    pdrs = []
    violations = 0
    rollbacks = 0
    admitted = 0
    proposals = 0
    rewards = []
    recovery_cycles = -1
    onset_cycle = scenario_cfg.get("onset", 0)

    with open(log_path, "w") if log_path else _NullWriter() as fp:
        for c in range(cycles):
            net.cycle = c
            net.step_environment(scenario_cfg)
            o = observe(net, last_action, scenario_cfg)
            a, decision = ctrl.step(o, scenario_cfg)
            if controller_name in ("llm_only", "rl", "shield_ric"):
                proposals += k
                if controller_name == "shield_ric":
                    if decision == "shield:fallback":
                        rollbacks += 1
                    else:
                        admitted += 1
                else:
                    admitted += 1
            v, vreason = is_violation(o, a)
            if v:
                violations += 1
            pdr = net.realised_pdr(a)
            pdrs.append(pdr)
            r = pdr - 0.02 * a.power_dbm - (1.0 if v else 0.0)
            rewards.append(r)
            if c > onset_cycle and recovery_cycles < 0 and pdr > 0.9:
                recovery_cycles = c - onset_cycle
            action_dict = {k: (str(val) if isinstance(val, np.generic) else val)
                           for k, val in asdict(a).items()}
            action_dict["poison"] = None if a.poison is None else list(a.poison)
            row = {
                "cycle": int(c),
                "sinr_obs": float(o.sinr_db),
                "interference_class": str(o.interference_class),
                "interference_strength": float(o.interference_strength),
                "backhaul_ok": bool(o.backhaul_ok),
                "trust_state": str(o.trust_state),
                "action": action_dict,
                "decision": str(decision),
                "pdr": float(pdr),
                "violation": bool(v),
                "violation_reason": str(vreason),
                "reward": float(r),
            }
            if log_path:
                fp.write(json.dumps(row) + "\n")
            last_action = a

    # crude p95 latency proxy: inverse PDR worst 5%
    pdr_arr = np.array(pdrs)
    p95_lat = float(1.0 / (np.percentile(pdr_arr, 5) + 1e-3))
    return RunSummary(
        scenario=scenario,
        controller=controller_name,
        seed=seed,
        cycles=cycles,
        mean_pdr=float(pdr_arr.mean()),
        p95_latency_proxy=p95_lat,
        safety_violations=int(violations),
        violation_rate=float(violations / cycles),
        recovery_cycles=recovery_cycles,
        rollbacks=int(rollbacks),
        admitted_actions=int(admitted),
        total_proposals=int(proposals),
        mean_reward=float(np.mean(rewards)),
    )


class _NullWriter:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def write(self, _): pass


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "narrowband": {"mode": "narrowband", "onset": 40, "end": 200},
    "wideband":   {"mode": "wideband",   "onset": 40, "end": 200},
    "bursty":     {"mode": "bursty",     "onset": 40, "end": 220},
    "backhaul":   {"mode": "backhaul",   "onset": 40, "end": 200},
    "compromised_xapp": {"mode": "compromised_xapp", "onset": 30, "end": 220},
    # Compound: wideband interference AND intermittent backhaul AND a compromised xApp.
    # No single hand-tuned heuristic is configured for this; the LLM planner has
    # to reason across symptoms and the shield has to filter the elevated poison rate.
    "compound":   {"mode": "compound", "onset": 30, "end": 220},
}


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def run_all(out_dir: Path, seeds: Iterable[int] = (0, 1, 2, 3, 4),
            controllers: Iterable[str] = tuple(CONTROLLERS.keys()),
            scenarios: Iterable[str] = tuple(SCENARIOS.keys()),
            cycles: int = 240) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    rows = []
    for sc in scenarios:
        for ct in controllers:
            for s in seeds:
                r = run(sc, ct, seed=s, cycles=cycles, results_dir=out_dir)
                rows.append(asdict(r))
                print(f"[done] {sc:>18s}  {ct:>10s}  s={s}  "
                      f"PDR={r.mean_pdr:.3f}  viol={r.violation_rate:.3f}  "
                      f"recovery={r.recovery_cycles}")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--cycles", type=int, default=240)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out = (here.parent / args.out).resolve()
    run_all(out_dir=out, seeds=args.seeds, cycles=args.cycles)


if __name__ == "__main__":
    main()
