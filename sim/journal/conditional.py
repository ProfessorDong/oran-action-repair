"""
Journal edition -- the conditional-mean reward r_bar(o, a).

Theorem 1 is stated for

    r_bar(o, a) = E[ r_t | o_t = o, a_t = a ],

i.e. the mean reward conditioned on what the CONTROLLER SEES, not on the
latent state.  Getting this right matters: if one instead benchmarks against
the latent state, the classifier's confusion is charged to the twin-error
constant xi and to the regret, both of which then blow up to their trivial
maxima.  That is what happens under the naive instantiation, where
xi ~ 1.0 made the bound vacuous.

The exogenous part of the latent state -- interference class, interference
strength, backhaul reachability, base SINR -- evolves independently of the
control action, so the conditional law P(latent | o) can be estimated once per
scenario by simulating the environment and the observation channel alone.  We
do that here and expose:

    ConditionalModel.r_bar(obs_key, action)     -- E[PDR | o, a]
    ConditionalModel.pi_dagger(obs_key, Phi)    -- argmax over the safe pool
    ConditionalModel.clairvoyant(latent, Phi)   -- the omniscient benchmark

The gap between pi_dagger and the clairvoyant optimum is the irreducible price
of imperfect perception and is reported separately in Section VII-E.

Author: Liang Dong.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from core import (
    Action, LatentState, Observation, SCENARIOS, SAFE_POOL, SAFE_FALLBACK,
    step_environment, observe, realised_pdr, ALL_ACTIONS, BASE_SINR_DB)



_PDR_VEC_CACHE: dict = {}
_MASK_CACHE: dict = {}


def _pdr_vector(st):
    """Noiseless delivered fraction of every action in A at one latent state."""
    key = latent_key(st)
    hit = _PDR_VEC_CACHE.get(key)
    if hit is None:
        hit = np.array([realised_pdr(st, a, noise=False) for a in ALL_ACTIONS()])
        _PDR_VEC_CACHE[key] = hit
    return hit


def _admissible_mask(state, o, phi):
    """Boolean mask over A of the Phi-admissible actions, keyed on the
    threshold crossings, which is all of o that admissibility depends on."""
    from shield import admissible
    key = (state, tuple(p.name for p in phi))
    hit = _MASK_CACHE.get(key)
    if hit is None:
        hit = np.array([admissible(o, a, phi) for a in ALL_ACTIONS()], dtype=bool)
        _MASK_CACHE[key] = hit
    return hit


STRENGTH_BUCKET = 0.10


def obs_key(o: Observation) -> tuple:
    """The statistic of the observation that the reward model conditions on:
    the PERCEIVED interference class, the measured interference power bucketed
    at 0.1, and backhaul reachability."""
    return (o.interference_class,
            round(o.interference_strength / STRENGTH_BUCKET) * STRENGTH_BUCKET,
            bool(o.backhaul_ok))


def latent_key(st: LatentState) -> tuple:
    return (st.interference_class, round(st.interference_strength, 2),
            bool(st.backhaul_ok), round(st.base_sinr_db, 2))


def _proto(key: tuple) -> LatentState:
    ic, s, bh, base = key
    st = LatentState(rng=np.random.default_rng(0))
    st.interference_class, st.interference_strength = ic, s
    st.backhaul_ok, st.base_sinr_db = bh, base
    return st


@dataclass
class ConditionalModel:
    scenario: str
    weights: dict            # obs_key -> {latent_key: probability}
    protos: dict             # latent_key -> LatentState
    _rbar_cache: dict = None
    _pi_cache: dict = None

    def __post_init__(self):
        self._rbar_cache = {}
        self._pi_cache = {}
        self._pi_full_cache = {}
        self._rbar_vec_cache = {}

    # -- E[PDR | o, a] ------------------------------------------------------
    def r_bar(self, ok: tuple, a: Action) -> float:
        ck = (ok, a.sig())
        hit = self._rbar_cache.get(ck)
        if hit is not None:
            return hit
        w = self.weights.get(ok)
        if not w:
            v = realised_pdr(_proto(("none", 0.0, True, BASE_SINR_DB)), a,
                             noise=False)
        else:
            v = sum(p * realised_pdr(self.protos[lk], a, noise=False)
                    for lk, p in w.items())
        self._rbar_cache[ck] = float(v)
        return float(v)

    # -- pi^dagger(o) = argmax over the Phi-admissible safe pool ------------
    def pi_dagger(self, ok: tuple, pool=SAFE_POOL) -> tuple[Action, float]:
        hit = self._pi_cache.get(ok)
        if hit is not None:
            return hit
        best_a, best_r = SAFE_FALLBACK, -1.0
        for a in pool:
            r = self.r_bar(ok, a)
            if r > best_r:
                best_a, best_r = a, r
        self._pi_cache[ok] = (best_a, float(best_r))
        return self._pi_cache[ok]

    def pi_dagger_full(self, ok: tuple, o, phi) -> tuple:
        """pi^dagger(o) = argmax over A_Phi(o) of r_bar(o, .), searching all of
        A rather than a template pool.

        Vectorized: the delivered fraction of every action at every latent
        prototype is precomputed once, and admissibility depends on o only
        through the four thresholds the observation-dependent predicates test,
        so there are at most sixteen distinct masks.  Filtering 20,160 actions
        per cycle in Python is what made the direct version unusable."""
        from core import THETA_Q, THETA_M, U_BAR
        state = (o.queue_priority > THETA_Q, o.queue_priority > THETA_M,
                 o.twin_uncertainty > U_BAR, bool(o.backhaul_ok))
        pk = (ok, state)
        hit = self._pi_full_cache.get(pk)
        if hit is not None:
            return hit
        mask = _admissible_mask(state, o, phi)
        rvec = self._rbar_vector(ok)
        if not mask.any():
            out = (SAFE_FALLBACK, self.r_bar(ok, SAFE_FALLBACK))
        else:
            idx = int(np.argmax(np.where(mask, rvec, -np.inf)))
            out = (ALL_ACTIONS()[idx], float(rvec[idx]))
        self._pi_full_cache[pk] = out
        return out

    def _rbar_vector(self, ok: tuple):
        hit = self._rbar_vec_cache.get(ok)
        if hit is not None:
            return hit
        w = self.weights.get(ok)
        if not w:
            v = _pdr_vector(_proto(("none", 0.0, True, BASE_SINR_DB)))
        else:
            v = np.zeros(len(ALL_ACTIONS()))
            for lk, pr in w.items():
                v = v + pr * _pdr_vector(self.protos[lk])
        self._rbar_vec_cache[ok] = v
        return v

    # -- omniscient benchmark ----------------------------------------------
    @staticmethod
    def clairvoyant(st, o=None, phi=None) -> tuple:
        """Best action at the TRUE latent state over the same admissible set
        pi_dagger searches, so their difference is the price of misperception
        and not of a different feasible set."""
        from core import THETA_Q, THETA_M, U_BAR
        pdr = _pdr_vector(st)
        if o is None or phi is None:
            idx = int(np.argmax(pdr))
            return ALL_ACTIONS()[idx], float(pdr[idx])
        state = (o.queue_priority > THETA_Q, o.queue_priority > THETA_M,
                 o.twin_uncertainty > U_BAR, bool(o.backhaul_ok))
        mask = _admissible_mask(state, o, phi)
        if not mask.any():
            return SAFE_FALLBACK, float(realised_pdr(st, SAFE_FALLBACK,
                                                     noise=False))
        idx = int(np.argmax(np.where(mask, pdr, -np.inf)))
        return ALL_ACTIONS()[idx], float(pdr[idx])


def build(scenario: str, confusion: dict | None, seeds: int = 60,
          cycles: int = 300) -> ConditionalModel:
    """Estimate P(latent | o) for one scenario by simulating the exogenous
    environment and the observation channel.  No controller is involved: the
    quantities that enter realised_pdr are all exogenous."""
    cfg = SCENARIOS[scenario]
    counts: dict = defaultdict(lambda: defaultdict(int))
    protos: dict = {}
    a_ref = Action()
    for sd in range(seeds):
        st = LatentState(rng=np.random.default_rng(100000 + sd))
        for c in range(cycles):
            st.cycle = c
            step_environment(st, cfg)
            o = observe(st, a_ref, cfg, confusion=confusion)
            lk = latent_key(st)
            counts[obs_key(o)][lk] += 1
            if lk not in protos:
                protos[lk] = _proto(lk)
    weights = {ok: {lk: n / sum(d.values()) for lk, n in d.items()}
               for ok, d in counts.items()}
    return ConditionalModel(scenario=scenario, weights=weights, protos=protos)


_CACHE: dict = {}


def get(scenario: str, confusion: dict | None, tag: str = "radioml"
        ) -> ConditionalModel:
    key = (scenario, tag)
    if key not in _CACHE:
        _CACHE[key] = build(scenario, confusion)
    return _CACHE[key]
