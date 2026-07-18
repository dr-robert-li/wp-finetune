#!/usr/bin/env python3
"""RC-A confirmation runner (Phase 04.4 / D-IT-02).

Re-runs the REVL-01A judge census on the EXISTING v3 staged merge
(models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v3) through the now-patched
eval_judge.run_eval (which passes chat_template_kwargs enable_thinking=False).

Single-variable test: the ONLY thing changed vs the original v3 census (parse 0.190,
Spearman 0.056) is the enable_thinking=False kwarg. If RC-A is correct, the merged-served
judge path should now reproduce the Tinker-runtime offline result E3 (0 parse failures,
overall Spearman 0.2626) — proving the adapter + weight merge are fine and the parse gate
that arrested plans 07/08 was harness-induced.

SUCCESS CRITERION (advisor-set, do not round up):
  RC-A CONFIRMED iff parse_failure_rate <= 0.05 AND overall_spearman ~= 0.2626 (|d| <= 0.03 vs E3).
  Parse recovers but Spearman materially below E3 -> PARTIAL (residual merged-weight-vs-runtime
  judge divergence), NOT full confirmation.

No new merge. No gen. No capture (the capture path is unrelated and previously hung).
Smoke (15 prompts) gates the full 121 census so a non-landing fix costs cents, not an hour.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V3_STAGING = ROOT / "models" / "_staging" / "qwen3-30b-wp-30_70-reasoning-merged-v3"
DATASET = "data/reasoning_dataset/openai_val.jsonl"
OUT_DIR = ROOT / "output" / "eval_reasoning_v3"
CONFIRM_OUT = OUT_DIR / "revl01a_v3_rcA_confirm.json"
JUDGE_OUT = OUT_DIR / "reasoning_v3_rcA_confirm" / "eval_judge_results.json"
LOG_DIR = ROOT / "logs" / "phase4.4"
LOG_PATH = LOG_DIR / "revl01a_v3_rcA_confirm.log"
PORT = 8021
GPU_MEM_UTIL = 0.55
CONTAINER_NAME = "wp-revl01a-v3-confirm-vllm"

E3_TINKER_SPEARMAN = 0.2626220775534824  # output/eval_reasoning/reasoning_v3_tinker/eval_judge_results.json
BASELINE_SPEARMAN = 0.2678275724901261   # merged-v2 baseline (revl01_baseline)
SPEARMAN_TOL = 0.03
SMOKE_N = 15
SMOKE_MAX_FAIL_RATE = 0.20  # if smoke is still this bad, fix did not land -> abort before the full hour


def _log(msg: str, fh) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def _census(endpoint, out_path, limit, fh):
    from eval import eval_judge
    jud = eval_judge.run_eval(
        dataset_path=DATASET,
        limit=limit,
        output_path=str(out_path),
        base_url=endpoint,
        output_format="auto",
        gt_mode="calibrated_canonical",
    )
    excluded = jud.get("excluded", {})
    parse_fail = excluded.get("parse_fail", 0)
    n = jud.get("n_examples", 0) or 0
    rate = (parse_fail / n) if n else 1.0
    sp = jud.get("revl01a_overall_spearman_HARD", {}) or {}
    return {"parse_fail": parse_fail, "n": n, "rate": rate,
            "spearman": sp.get("corr"), "n_pairs": sp.get("n_pairs"),
            "p_value": sp.get("p_value"), "excluded": excluded}


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    JUDGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as fh:
        if not (V3_STAGING / "config.json").exists():
            _log(f"ABORT: v3 staging dir missing: {V3_STAGING}", fh)
            return 1
        _log(f"v3 staging: {V3_STAGING}", fh)
        _log(f"E3 reference Spearman (Tinker-runtime): {E3_TINKER_SPEARMAN:.4f}", fh)

        from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm, VllmBootTimeout

        endpoint = f"http://localhost:{PORT}/v1"
        os.environ["EVAL_JUDGE_BASE_URL"] = endpoint

        result = {"measured_on": "merged-served-v3", "model_path": str(V3_STAGING),
                  "fix": "eval_judge enable_thinking=False (RC-A)", "val_set": DATASET,
                  "e3_tinker_spearman": E3_TINKER_SPEARMAN, "baseline_spearman": BASELINE_SPEARMAN}
        try:
            _log(f"Booting vLLM: {CONTAINER_NAME} model={V3_STAGING} port={PORT}", fh)
            boot_vllm(str(V3_STAGING), CONTAINER_NAME, PORT, GPU_MEM_UTIL)
            served = wait_healthy(PORT, CONTAINER_NAME)
            _log(f"vLLM healthy; served model: {served}", fh)

            # ---- SMOKE (cheap de-risk before the full hour) ----
            _log(f"SMOKE: {SMOKE_N}-prompt census ...", fh)
            smoke = _census(endpoint, JUDGE_OUT.with_name("smoke_judge_results.json"), SMOKE_N, fh)
            _log(f"SMOKE: parse_fail={smoke['parse_fail']}/{smoke['n']} rate={smoke['rate']:.4f} "
                 f"spearman={smoke['spearman']}", fh)
            result["smoke"] = smoke
            if smoke["rate"] > SMOKE_MAX_FAIL_RATE:
                _log(f"ABORT: smoke parse rate {smoke['rate']:.4f} > {SMOKE_MAX_FAIL_RATE} — RC-A fix "
                     f"did NOT land on this path (or template rejected the kwarg — check WARNING above). "
                     f"Not spending the full census.", fh)
                result["rc_a_confirmed"] = False
                result["verdict"] = "smoke_abort_fix_did_not_land"
                CONFIRM_OUT.write_text(json.dumps(result, indent=2))
                return 2

            # ---- FULL 121 census ----
            _log("FULL: 121-row census ...", fh)
            full = _census(endpoint, JUDGE_OUT, None, fh)
            _log(f"FULL: parse_fail={full['parse_fail']}/{full['n']} rate={full['rate']:.4f} "
                 f"spearman={full['spearman']} n_pairs={full['n_pairs']}", fh)
            result["full"] = full

            rate = full["rate"]
            sp = full["spearman"]
            parse_gate_pass = rate <= 0.05
            spearman_ok = (sp is not None) and (abs(sp - E3_TINKER_SPEARMAN) <= SPEARMAN_TOL)
            rc_a_confirmed = parse_gate_pass and spearman_ok

            result.update({
                "parse_failure_rate": rate,
                "parse_gate_pass": parse_gate_pass,
                "overall_spearman": sp,
                "spearman_matches_e3": spearman_ok,
                "spearman_delta_vs_e3": (None if sp is None else sp - E3_TINKER_SPEARMAN),
                "rc_a_confirmed": rc_a_confirmed,
                "v3_original_parse_rate": 23 / 121,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            if rc_a_confirmed:
                result["verdict"] = "RC-A CONFIRMED: parse recovered AND Spearman ~= E3"
            elif parse_gate_pass:
                result["verdict"] = ("PARTIAL: parse recovered but Spearman below E3 — residual "
                                     "merged-weight-vs-Tinker-runtime judge divergence")
            else:
                result["verdict"] = "NOT CONFIRMED: parse rate still > 0.05 after fix"
            _log(f"VERDICT: {result['verdict']}", fh)
            CONFIRM_OUT.write_text(json.dumps(result, indent=2))
            _log(f"Wrote: {CONFIRM_OUT}", fh)
            return 0 if rc_a_confirmed else 3
        except VllmBootTimeout as e:
            _log(f"ABORT: vLLM boot timeout: {e}", fh)
            return 1
        finally:
            stop_vllm(CONTAINER_NAME)
            _log("vLLM stopped.", fh)


if __name__ == "__main__":
    raise SystemExit(main())
