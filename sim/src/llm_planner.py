"""
Real local-LLM planner for SHIELD-RIC.

Replaces the deterministic LLMPlannerStub with a HuggingFace Transformers
backend (default: Qwen2.5-1.5B-Instruct, small enough to run on an 8 GB GPU
in fp16).  The planner emits a JSON list of K candidate actions per cycle.

Two design goals:

    1. The LLM never controls the radio directly.  It only emits candidate
       actions that are then filtered by the SHIELD-RIC safety shield.  By
       Lemma~\\ref{lem:soundness} this swap preserves the soundness theorem
       regardless of which model is used.

    2. The LLM is asked to produce STRUCTURED JSON.  Out-of-grammar outputs
       are caught by a JSON parser and replaced with safe fallbacks; we
       count these as "malformed".

We report, per scenario:  Lemma 1 holds (violation rate stays 0.000),
mean cov_K of the LLM planner (vs. stub), mean PDR, and end-to-end timing.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import shield_ric as sr
from shield_ric import (
    Action, Observation, NetworkState, observe, twin_predict,
    is_violation, SCENARIOS, ShieldConfig, shield_admit_set,
    WAVEFORMS, MOD_CODES, BEAMS, ROUTES, SLICES, WORKLOADS,
)
from verify_theorem import (
    safe_action_pool, oracle_best_action, action_sig, SAFE_POOL,
)


HERE = Path(__file__).resolve().parent
SIM  = HERE.parent
RESULTS = SIM / "results"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an Open RAN control planner.  At each cycle you read
a structured network observation and emit exactly K candidate cross-layer
recovery actions as JSON.  You do NOT actuate; a downstream safety shield
checks every candidate.

Each candidate action has fields:
  waveform   : one of [ofdm_n0, ofdm_n1, ofdm_n2, dsss]
  mod_code   : one of [qpsk_1_3, qpsk_1_2, 16qam_1_2, 16qam_3_4, 64qam_3_4]
  beam       : one of [omni, sector_a, sector_b, narrow]
  route      : one of [primary, secondary, mesh_a, mesh_b]
  slice_     : one of [urgent, balanced, best_effort]
  workload   : one of [local, edge, cloud]
  power_dbm  : a number in [10, 30]

Reply with EXACTLY one JSON object:
  {"candidates": [ { ... }, { ... }, ... ]}   with K candidates.
No prose, no markdown, no leading text.  K = 5.
"""


def obs_to_prompt(o: Observation, scenario_cfg: dict) -> str:
    return (
        f"Network observation (cycle {int(o.t)}):\n"
        f"  interference_class: {o.interference_class}\n"
        f"  interference_strength: {o.interference_strength:.2f}\n"
        f"  sinr_db: {o.sinr_db:.2f}\n"
        f"  pdr_recent: {o.pdr:.2f}\n"
        f"  backhaul_ok: {o.backhaul_ok}\n"
        f"  queue_priority: {o.queue_priority:.2f}\n"
        f"  twin_uncertainty: {o.twin_uncertainty:.2f}\n"
        f"  trust_state: {o.trust_state}\n"
        "\n"
        "Mission intent: preserve priority-flow PDR > 0.9 under "
        "current interference while minimizing transmit power and "
        "respecting all spectrum / identity / QoS policies."
    )


# ---------------------------------------------------------------------------
# LLM-driven planner
# ---------------------------------------------------------------------------

class LLMPlanner:
    """Real local-LLM planner that emits K candidate Actions as JSON."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str | None = None, k: int = 5,
                 max_new_tokens: int = 480, temperature: float = 0.6,
                 safe_fallback: Action | None = None):
        self.k = k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.safe_fallback = safe_fallback or Action()
        self.n_malformed = 0
        self.n_calls = 0
        self.total_call_seconds = 0.0

    @torch.no_grad()
    def propose(self, o: Observation, scenario_cfg: dict) -> list[Action]:
        user = obs_to_prompt(o, scenario_cfg)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        t0 = time.time()
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0
        self.total_call_seconds += elapsed
        self.n_calls += 1
        gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
        raw = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return self._parse(raw)

    @staticmethod
    def _pick(field: str, value, allowed: tuple, default):
        s = str(value)
        return s if s in allowed else default

    def _parse(self, raw: str) -> list[Action]:
        try:
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                raise ValueError("no json object")
            data = json.loads(m.group(0))
            cands = data.get("candidates", [])
            actions: list[Action] = []
            for c in cands[: self.k]:
                try:
                    power = float(c.get("power_dbm", 23.0))
                except (TypeError, ValueError):
                    power = 23.0
                a = Action(
                    waveform=self._pick("waveform", c.get("waveform", "ofdm_n0"),
                                        WAVEFORMS, "ofdm_n0"),
                    mod_code=self._pick("mod_code", c.get("mod_code", "qpsk_1_2"),
                                        MOD_CODES, "qpsk_1_2"),
                    beam=self._pick("beam", c.get("beam", "omni"),
                                    BEAMS, "omni"),
                    route=self._pick("route", c.get("route", "primary"),
                                     ROUTES, "primary"),
                    slice_=self._pick("slice_", c.get("slice_", "balanced"),
                                      SLICES, "balanced"),
                    workload=self._pick("workload", c.get("workload", "local"),
                                        WORKLOADS, "local"),
                    power_dbm=float(np.clip(power, 10.0, 40.0)),
                )
                actions.append(a)
            while len(actions) < self.k:
                actions.append(self.safe_fallback)
            return actions
        except Exception:
            self.n_malformed += 1
            # Malformed JSON -> use safe fallback for all candidates.
            return [self.safe_fallback for _ in range(self.k)]


# ---------------------------------------------------------------------------
# Driver that runs a SHIELD-RIC loop with the LLM planner
# ---------------------------------------------------------------------------

def llm_run(scenario: str, seed: int, cycles: int = 100,
            perception_confusion: dict | None = None,
            model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
            ) -> dict:
    rng = np.random.default_rng(seed)
    net = NetworkState(rng=rng)
    scenario_cfg = SCENARIOS[scenario]
    shield_cfg = ShieldConfig()
    sr.PERCEPTION_CONFUSION = perception_confusion
    planner = LLMPlanner(model_id=model_id)

    last_action = Action()
    fallback = Action()

    pdrs, violations, regrets = [], 0, []
    cov_hits, n_total = 0, 0
    rollbacks, admitted = 0, 0

    for c in range(cycles):
        net.cycle = c
        net.step_environment(scenario_cfg)
        o = observe(net, last_action, scenario_cfg)
        cands = planner.propose(o, scenario_cfg)
        admit = shield_admit_set(o, cands, shield_cfg)
        if admit:
            chosen = max(admit, key=lambda t: t[1][0])[0]
            admitted += 1
        else:
            chosen = fallback
            rollbacks += 1
        v, _ = is_violation(o, chosen)
        if v: violations += 1
        pdr = net.realised_pdr(chosen)
        true_r = pdr - 0.02 * chosen.power_dbm - (1.0 if v else 0.0)
        oracle_a, oracle_r = oracle_best_action(net, o)
        sig_set = {action_sig(a) for a in cands}
        cov_hits += int(action_sig(oracle_a) in sig_set)
        regrets.append(max(0.0, oracle_r - true_r))
        pdrs.append(pdr)
        n_total += 1
        last_action = chosen

    return {
        "scenario": scenario,
        "seed": seed,
        "cycles": cycles,
        "mean_pdr": float(np.mean(pdrs)),
        "violation_rate": float(violations / n_total),
        "cov_K_mean": float(cov_hits / n_total),
        "regret_per_cycle": float(np.mean(regrets)),
        "rollback_rate": float(rollbacks / n_total),
        "n_malformed_outputs": int(planner.n_malformed),
        "n_llm_calls": int(planner.n_calls),
        "mean_call_seconds": float(planner.total_call_seconds / max(planner.n_calls, 1)),
        "model_id": model_id,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=80,
                    help="cycles per scenario (kept small for LLM time)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--scenarios", nargs="+",
                    default=["narrowband", "wideband", "compromised_xapp"])
    ap.add_argument("--model",
                    default=str((SIM / "models/Qwen2.5-1.5B-Instruct").resolve()))
    ap.add_argument("--perception", choices=["perfect", "radioml"],
                    default="radioml")
    ap.add_argument("--out", default=str(RESULTS / "llm_planner.csv"))
    args = ap.parse_args()

    conf = None
    if args.perception == "radioml":
        p = RESULTS / "perception_radioml.json"
        if p.exists():
            conf = json.loads(p.read_text())["perception_confusion"]

    rows = []
    for sc in args.scenarios:
        for s in args.seeds:
            print(f"[llm] scenario={sc}, seed={s}, model={args.model}")
            res = llm_run(sc, s, cycles=args.cycles,
                          perception_confusion=conf, model_id=args.model)
            rows.append(res)
            print(f"  -> PDR={res['mean_pdr']:.3f}  viol={res['violation_rate']:.3f}  "
                  f"cov_K={res['cov_K_mean']:.3f}  malformed={res['n_malformed_outputs']}"
                  f"  call_s={res['mean_call_seconds']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\nAggregate:")
    print(df.groupby("scenario").agg(
        mean_pdr=("mean_pdr","mean"),
        violation_rate=("violation_rate","mean"),
        cov_K_mean=("cov_K_mean","mean"),
        regret_per_cycle=("regret_per_cycle","mean"),
        n_malformed=("n_malformed_outputs","mean"),
        mean_call_seconds=("mean_call_seconds","mean"),
    ).round(3).to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
