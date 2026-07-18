#!/usr/bin/env python
"""v4 judge MoE-Sieve k-sweep driver — JUDGE-ONLY (Plan 25-02, GATE4-03).

Adapts scripts/sieve_ksweep_run.py (v3) to the v4 judge (Qwen3.6-35B-A3B, 256
experts) and drops the gen/wp-bench arm entirely: this is a judge-only artifact.

For each k in the LOCKED pre-registered grid (read verbatim from
output/sieve-v4/ksweep_preregistration.md), build the per-layer keep-mask
(top-k hot experts by this profile's total counts UNION the protected set), serve
the v4 judge s1 through the patched vLLM (scripts/_sieve_vllm_patch, -inf keep-mask),
capture 121 wp_judge prompts @ max_tokens=8192 (truncation-aware; the 2048 default
is the carry-forward #1 false-negative), score single-seed s1 Spearman rho vs
val_labels_v1, and record it into output/sieve-v4/k_sweep_results_v4.json.

Arm order: `full` FIRST — it does triple duty:
  1. LIVE patch confirmation: booted with an ALL-KEEP [40,256] mask so the
     _sieve_vllm_patch installs and resolves the qwen3_5_moe/qwen3_next
     SparseMoeBlock class (or FAILS LOUD). It cannot silently serve unmasked while
     a mask env is set. All-keep => quality is identical to unmasked.
  2. Same-stack TOST reference: its s1 rho is what every masked arm is TOST'd
     against (NOT the llama.cpp Q8 0.8067 nor Tinker 0.8358).
  3. Sanity floor: full-arm rho below FULL_JUDGE_RHO_FLOOR => HALT (harness
     misconfig, not a masking result).

One 35B residency at a time (GB10 serialization): arms run sequentially, one
vLLM container up at a time. Resume support: arms already in the results json are
kept, not re-run.

ENSEMBLE (reserved): if a masked k passes CI-aware TOST vs the full arm AND retains
protected experts, that ONE k is re-served with s0 and s2 and the 3-seed
median-ensemble rho recorded as confirmation. Not run at every k.

Usage (run as ONE backgrounded driver):
    nohup .venv-tinker/bin/python scripts/sieve_ksweep_v4_run.py \
        > logs/sieve-v4/ksweep_driver.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm  # noqa: E402
from scripts.sieve_expert_mask_inference import (  # noqa: E402
    build_ksweep_mask, build_profile_counts, load_protected_mask,
)
from scripts.sieve_v4_tost_verdict import (  # noqa: E402
    load_labels, score_capture, tost_from_scores,
)

# --- v4 judge config ---------------------------------------------------------
JUDGE_SEEDS = {
    "s0": "models/Qwen3.6-35B-A3B-judge-v4-s0-merged",
    "s1": "models/Qwen3.6-35B-A3B-judge-v4-s1-merged",
    "s2": "models/Qwen3.6-35B-A3B-judge-v4-s2-merged",
}
PROFILE_ROUTING = "output/sieve-v4/routing_report.jsonl"   # 25-01 served-model profile
PROTECTED_MASK = "output/sieve-v4/protected_expert_mask.npy"  # 25-01, [40,256] bool
PREREG = "output/sieve-v4/ksweep_preregistration.md"       # LOCKED grid source

VAL_DATASET = "data/reasoning_dataset/openai_val.jsonl"    # 121 wp_judge (Wave-2 eval set)
VAL_LABELS = "output/relabel/val_labels_v1.json"

# Serve pattern: v4 judge is a VL checkpoint -> --language-model-only; patched
# vLLM reads SIEVE_MASK_NPY (host) -> mounts _sieve_vllm_patch + sets
# SIEVE_KEEP_MASK_NPY inside the container. Weights ~67 GiB; give KV headroom.
PORT = 8021
GPU_MEM_UTIL = 0.85
MAX_MODEL_LEN = 16384          # max judge prompt ~2.2K tokens + 8192 generation budget
CAPTURE_MAX_TOKENS = 8192      # carry-forward #1: NOT the 2048 default
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_30_70_vllm.sh")

MASK_DIR = PROJECT_ROOT / "output/sieve-v4/masks"
OUT_ROOT = PROJECT_ROOT / "output/sieve-v4/ksweep"
RESULTS_PATH = PROJECT_ROOT / "output/sieve-v4/k_sweep_results_v4.json"

# Full-arm sanity floor (pre-registration §4: bf16-vLLM-served s1 ~0.7872 anchor,
# floor ~0.72). Below floor => HALT the sweep (harness misconfig).
FULL_JUDGE_RHO_FLOOR = 0.72

_ALL_KEEP_MASK = MASK_DIR / "full_all_keep.npy"  # full arm: proves the patch is live


def parse_grid(prereg_path: Path) -> list[int]:
    """Read the LOCKED grid verbatim from the committed pre-registration.

    Parses the '**Grid: { full (256), 224, 192, 144, 112 }**' line. The masked ks
    are every integer in the grid except the 256 'full' control. The grid is NEVER
    altered after a rho is read (threat T-25-06).
    """
    text = prereg_path.read_text()
    m = re.search(r"Grid:\s*\{([^}]*)\}", text)
    if not m:
        raise SystemExit(f"could not find the locked grid in {prereg_path}")
    ints = [int(x) for x in re.findall(r"\d+", m.group(1))]
    # drop 'full' == n_experts (256); descending order for the sweep
    masked = sorted({i for i in ints if i != 256}, reverse=True)
    if not masked:
        raise SystemExit(f"no masked ks parsed from grid: {ints}")
    return masked


def _set_mask_env(mask_path: Path | None) -> None:
    if mask_path is None:
        os.environ.pop("SIEVE_MASK_NPY", None)
    else:
        os.environ["SIEVE_MASK_NPY"] = str(mask_path)


def build_keep_mask_for_k(k: int, counts: np.ndarray, protected: np.ndarray) -> tuple[Path, dict]:
    keep = build_ksweep_mask(counts, protected, k)
    path = MASK_DIR / f"keep_k{k}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, keep)
    protected_retained = bool(np.all((~protected) | keep))
    meta = {
        "kept_experts_per_layer": keep.sum(axis=1).astype(int).tolist(),
        "protected_retained": protected_retained,
    }
    return path, meta


def _ensure_all_keep_mask(n_layers: int, n_experts: int) -> Path:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_ALL_KEEP_MASK, np.ones((n_layers, n_experts), dtype=bool))
    return _ALL_KEEP_MASK


def capture_seed(k, seed: str, mask_path: Path) -> Path:
    """Boot one v4 judge seed with the given keep-mask, capture 121 @8192, stop."""
    from scripts.sieve_capture_judge_http import capture

    name = f"sieve-v4-judge-{seed}-k{k}"
    out = OUT_ROOT / f"k{k}" / seed / "judge_responses.jsonl"
    _set_mask_env(mask_path)
    try:
        boot_vllm(JUDGE_SEEDS[seed], name, PORT, GPU_MEM_UTIL,
                  serve_script=SERVE_SCRIPT,
                  extra_env={"LANGUAGE_MODEL_ONLY": "1", "MAX_MODEL_LEN": str(MAX_MODEL_LEN)})
        wait_healthy(PORT, name)
        capture(base_url=f"http://localhost:{PORT}/v1", model=None,
                dataset=str(PROJECT_ROOT / VAL_DATASET), out=str(out),
                max_tokens=CAPTURE_MAX_TOKENS, temperature=0.0)
    finally:
        _set_mask_env(None)
        stop_vllm(name)
    return out


def score_s1(capture_path: Path, labels: dict) -> dict:
    from scipy.stats import spearmanr
    scores, parse_fail = score_capture(capture_path)
    common = sorted(set(scores) & set(labels))
    rho = (float(spearmanr([scores[k] for k in common],
                           [labels[k] for k in common]).statistic)
           if len(common) > 2 else None)
    return {"judge_single_s1_rho": rho, "parse_fail": parse_fail, "n_scored": len(common)}


def run_full_arm(counts: np.ndarray, protected: np.ndarray, labels: dict) -> dict:
    n_layers, n_experts = counts.shape
    print("=== k=full: LIVE patch confirmation via all-keep mask (fail-loud) ===", flush=True)
    all_keep = _ensure_all_keep_mask(n_layers, n_experts)
    cap = capture_seed("full", "s1", all_keep)
    res = score_s1(cap, labels)
    return {
        "k": "full",
        "judge_capture": str(cap.relative_to(PROJECT_ROOT)),
        "kept_experts_per_layer": [n_experts] * n_layers,
        "protected_retained": True,
        "masked": False,
        "patch_confirmation": "booted with all-keep mask; see boot log for '[sieve-mask] patched ...'",
        **res,
    }


def run_masked_arm(k: int, counts: np.ndarray, protected: np.ndarray, labels: dict) -> dict:
    print(f"=== k={k}: building keep-mask + serving s1 ===", flush=True)
    mask_path, meta = build_keep_mask_for_k(k, counts, protected)
    cap = capture_seed(k, "s1", mask_path)
    res = score_s1(cap, labels)
    return {
        "k": str(k),
        "judge_capture": str(cap.relative_to(PROJECT_ROOT)),
        "masked": True,
        **meta,
        **res,
    }


def maybe_run_ensemble(arm: dict, full_arm: dict, k: int, counts: np.ndarray,
                       protected: np.ndarray, labels: dict) -> dict | None:
    """Reserved 3-seed confirmation: only when the masked s1 arm passes CI-aware
    TOST vs the full arm AND retains protected experts."""
    if not arm.get("protected_retained"):
        return None
    full_scores, _ = score_capture(PROJECT_ROOT / full_arm["judge_capture"])
    masked_scores, _ = score_capture(PROJECT_ROOT / arm["judge_capture"])
    tost = tost_from_scores(masked_scores, full_scores, labels)
    arm["s1_tost_vs_full"] = tost
    if not tost.get("equivalent"):
        print(f"=== k={k}: s1 not TOST-equivalent (ci={tost['ci']}) — ensemble skipped ===", flush=True)
        return None
    print(f"=== k={k}: s1 TOST-EQUIVALENT — running 3-seed ensemble confirmation ===", flush=True)
    from scipy.stats import spearmanr
    mask_path = MASK_DIR / f"keep_k{k}.npy"
    per_seed = {"s1": masked_scores}
    for seed in ("s0", "s2"):
        cap = capture_seed(k, seed, mask_path)
        sc, _ = score_capture(cap)
        per_seed[seed] = sc
    ensemble = {}
    for key in labels:
        vals = [per_seed[s][key] for s in per_seed if key in per_seed[s]]
        if vals:
            ensemble[key] = float(np.median(vals))
    common = sorted(set(ensemble) & set(labels))
    ens_rho = (float(spearmanr([ensemble[k2] for k2 in common],
                               [labels[k2] for k2 in common]).statistic)
               if len(common) > 2 else None)
    return {"judge_ensemble_rho": ens_rho, "n_scored": len(common)}


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    grid = parse_grid(PROJECT_ROOT / PREREG)
    print(f"[grid] locked masked ks (descending): {grid}", flush=True)

    counts = build_profile_counts([PROJECT_ROOT / PROFILE_ROUTING])
    protected = load_protected_mask(PROJECT_ROOT / PROTECTED_MASK)
    assert counts.shape == protected.shape, f"{counts.shape} != {protected.shape}"
    labels = load_labels()

    # Resume support: keep arms already recorded.
    sweep = []
    if RESULTS_PATH.exists():
        sweep = json.loads(RESULTS_PATH.read_text()).get("sweep", [])
    done_ks = {r["k"] for r in sweep}

    def persist(halted=False, halt_reason=None):
        payload = {"sweep": sweep, "grid": grid, "halted": halted}
        if halt_reason:
            payload["halt_reason"] = halt_reason
        RESULTS_PATH.write_text(json.dumps(payload, indent=2))

    order = ["full", *grid]
    full_arm = next((r for r in sweep if r["k"] == "full"), None)

    for k in order:
        key = "full" if k == "full" else str(k)
        if key in done_ks:
            arm = next(r for r in sweep if r["k"] == key)
            print(f"=== k={k}: already recorded (s1_rho={arm.get('judge_single_s1_rho')}), skipping ===",
                  flush=True)
            if k == "full":
                full_arm = arm
        else:
            t0 = time.time()
            if k == "full":
                arm = run_full_arm(counts, protected, labels)
                full_arm = arm
            else:
                arm = run_masked_arm(k, counts, protected, labels)
                ens = maybe_run_ensemble(arm, full_arm, k, counts, protected, labels)
                if ens:
                    arm.update(ens)
            arm["duration_sec"] = round(time.time() - t0, 1)
            sweep.append(arm)
            persist()

        print(f"=== k={k} DONE: s1_rho={arm.get('judge_single_s1_rho')} "
              f"parse_fail={arm.get('parse_fail')} "
              f"protected_retained={arm.get('protected_retained')} "
              f"({arm.get('duration_sec')}s) ===", flush=True)

        if k == "full":
            rho = arm.get("judge_single_s1_rho")
            if rho is None or rho < FULL_JUDGE_RHO_FLOOR:
                reason = (f"full-arm s1 rho={rho} below floor {FULL_JUDGE_RHO_FLOOR} — "
                          "harness misconfiguration, not a masking result")
                persist(halted=True, halt_reason=reason)
                print(f"HALT: {reason}. Aborting sweep.", flush=True)
                return 1

    persist()
    print("=== k-sweep COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
