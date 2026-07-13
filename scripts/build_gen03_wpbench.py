#!/usr/bin/env python
"""Phase 21 Plan 05 Task 2 -- GEN-03 wp-bench: serve the merged gen model,
run the full 344-test wp-bench suite via the exact reused
scripts.run_eval_reasoning._run_wpbench harness (enable_thinking=False,
max_tokens=2048, concurrency=4, seed 1337, request_timeout 1800s), and
compare the CI-aware bootstrap lower bound against the pre-registered
0.4286 floor.

Reuses the Phase 15 LOCKED lesson (real-generation warm-up gate, not a
/v1/models health check) exactly as scripts/bench_wpbench_base_anchor.py
does. Stops the serve in a finally block (sole-GB10-residency discipline).

If the CI lower bound misses the inherited 0.4286 floor, runs a fresh raw
new-base anchor (same harness, RAW models/Qwen3.6-35B-A3B, no adapter) to
check whether the new base's OWN raw coding ability has materially shifted
the noise band before considering any floor swap -- per plan instruction,
a miss is a recorded, valid outcome; floors are never silently swapped
without this measured anchor.

Must run under the Tinker venv (openai client + boot/wait/generate/stop
helpers only need `openai`, already present there; vLLM itself runs in the
docker serve stack, not this venv):
    .venv-tinker/bin/python scripts/build_gen03_wpbench.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm, VllmBootTimeout  # noqa: E402
import scripts.run_eval_reasoning as rer  # noqa: E402

MERGE_RECEIPT = PROJECT_ROOT / "output" / "base21" / "gen03_merge.json"
OUT_DIR = PROJECT_ROOT / "output" / "base21"
OUT_PATH = OUT_DIR / "gen03_wpbench.json"
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
PORT = 8024
GPU_MEM_UTIL = 0.80
INHERITED_FLOOR = 0.4286
N_BOOT = 1000
ALPHA = 0.05
# D-09 seed-noise-floor reference (output/sieve/optimal_k.json) -- the bar
# for treating a raw new-base anchor as evidence of a MATERIAL noise-band
# shift (not just measurement noise) vs the carried v3.0 floor.
_optimal_k = json.loads((PROJECT_ROOT / "output" / "sieve" / "optimal_k.json").read_text())
SEED_NOISE_FLOOR = _optimal_k["seed_noise_floor"]


def _real_generation_warmup(served: str) -> None:
    warm = generate(PORT, served,
                    [{"instruction": "Reply with exactly one word: OK", "source_val_idx": "warmup"}],
                    max_tokens=16)
    if not warm or not warm[0].strip():
        raise RuntimeError(f"Real-generation warm-up returned empty output: {warm!r}")
    print(f"[warmup] real-generation OK (served_model={served!r}): {warm[0].strip()[:80]!r}",
          file=sys.stderr)


def _run_wpbench_on(model_dir: str, container: str, tag: str) -> dict:
    """Boot vLLM (LANGUAGE_MODEL_ONLY), warm-up gate, run the reused
    _run_wpbench harness, stop in a finally block."""
    t0 = time.time()
    try:
        # SERVED_MODEL_NAME=wp-30_70: _run_wpbench's litellm config hardcodes
        # that model name (the identity serve_30_70_vllm.sh always sets);
        # serve_base20_vllm.sh otherwise serves as /workspace/model and every
        # bench request 404s ("The model `wp-30_70` does not exist").
        boot_vllm(model_dir, container, PORT, GPU_MEM_UTIL,
                  serve_script=SERVE_SCRIPT,
                  extra_env={"LANGUAGE_MODEL_ONLY": "1", "SERVED_MODEL_NAME": "wp-30_70"})
        served = wait_healthy(PORT, container, timeout=1200)
        _real_generation_warmup(served)
        rer.PORT = PORT
        result = rer._run_wpbench(tag, OUT_DIR)
    finally:
        stop_vllm(container)
    result["served_model_dir"] = model_dir
    result["wall_clock_s"] = round(time.time() - t0, 1)
    return result


def _wp_bench_overall(knowledge_mean: float | None, correctness_mean: float | None,
                       quality_mean: float | None) -> float:
    """Mirrors wp_bench.scoring.ScoreBreakdown.overall(): weighted average of
    ONLY the active (non-None) components, weights {knowledge:0.3,
    correctness:0.4, quality:0.3}."""
    weights = {"knowledge": 0.3, "correctness": 0.4, "quality": 0.3}
    values = {"knowledge": knowledge_mean, "correctness": correctness_mean, "quality": quality_mean}
    active = {k: w for k, w in weights.items() if values[k] is not None}
    if not active:
        return 0.0
    total_weight = sum(active.values())
    total = sum(values[k] * w for k, w in active.items())
    return round(total / total_weight, 4)


def _bootstrap_ci_lower(results_json_path: Path, n_boot: int = N_BOOT, alpha: float = ALPHA) -> dict:
    """Stratified bootstrap over wp-bench's own per-test-type breakdown
    (knowledge: per-test `score`; execution: per-test `correctness`),
    recombined via the exact wp_bench.scoring weighted-overall formula each
    resample -- this reproduces the actual "overall" statistic's sampling
    distribution (a flat unweighted per-test bootstrap would NOT, since
    overall is a 0.3/0.4/0.3-weighted combination of unequal-size strata,
    not a simple mean across all 344 tests). Percentile method, same
    convention as scripts/compute_concentration.py::bootstrap_ci."""
    data = json.loads(results_json_path.read_text())
    results = data["results"]
    knowledge = np.array([r["score"] for r in results if r.get("type") == "knowledge"], dtype=float)
    correctness = np.array([r["correctness"] for r in results if r.get("type") == "execution"], dtype=float)
    quality = np.array([r["quality"] for r in results
                        if r.get("type") == "execution" and r.get("quality") is not None], dtype=float)
    quality_mean = float(quality.mean()) if quality.size else None

    point = _wp_bench_overall(
        float(knowledge.mean()) if knowledge.size else None,
        float(correctness.mean()) if correctness.size else None,
        quality_mean,
    )

    rng = np.random.default_rng()
    boot_overall = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        k_resample = rng.choice(knowledge, size=knowledge.size, replace=True) if knowledge.size else knowledge
        c_resample = rng.choice(correctness, size=correctness.size, replace=True) if correctness.size else correctness
        boot_overall[i] = _wp_bench_overall(
            float(k_resample.mean()) if k_resample.size else None,
            float(c_resample.mean()) if c_resample.size else None,
            quality_mean,  # quality held fixed (no per-test quality signal here; null across the run)
        )

    lo = float(np.percentile(boot_overall, 100 * alpha / 2))
    hi = float(np.percentile(boot_overall, 100 * (1 - alpha / 2)))
    return {
        "point": point,
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "n_knowledge": int(knowledge.size),
        "n_execution": int(correctness.size),
        "n_boot": n_boot,
        "alpha": alpha,
    }


def _newest_results_json(tag: str) -> Path:
    cands = sorted((OUT_DIR / tag).glob("wp_bench_results_*.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise RuntimeError(f"no wp_bench_results_*.json found under {OUT_DIR / tag}")
    return cands[-1]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merge_receipt = json.loads(MERGE_RECEIPT.read_text())
    if not merge_receipt.get("merge_ok") or not merge_receipt.get("base_vs_merged_differs"):
        print(f"HALT: {MERGE_RECEIPT} does not report a clean merge (merge_ok/base_vs_merged_differs) "
              f"-- refusing to bench an unverified merge.", file=sys.stderr)
        return 2
    merged_dir = merge_receipt["merged_dir"]

    t0 = time.time()
    wp = _run_wpbench_on(merged_dir, "gen03-wpbench-vllm", "gen03_full")
    # _run_wpbench returns ran=True even when the wp-bench subprocess exits
    # non-zero (error captured in its dict + wp_bench_run.log) -- check the
    # score too, and surface the harness's own error instead of crashing later
    # in the results-file glob.
    if not wp.get("ran") or wp.get("wpbench_score") is None:
        print(f"HALT: wp-bench failed: {json.dumps(wp, indent=2)[:3000]}", file=sys.stderr)
        (OUT_PATH).write_text(json.dumps(wp, indent=2))
        return 3

    results_json = _newest_results_json("gen03_full")
    ci = _bootstrap_ci_lower(results_json)

    floor = INHERITED_FLOOR
    floor_source = "inherited"
    fresh_anchor = None
    pass_ = ci["ci_lower"] >= floor

    if not pass_:
        print("[gen03] CI lower bound misses the inherited floor -- running a fresh raw "
              "new-base anchor to check for a material noise-band shift before any floor "
              "decision (per plan: never silently swap floors without a measured anchor).",
              file=sys.stderr)
        anchor_wp = _run_wpbench_on("models/Qwen3.6-35B-A3B", "gen03-base-anchor-vllm", "gen03_fresh_new_base_anchor")
        anchor_results_json = (_newest_results_json("gen03_fresh_new_base_anchor")
                               if anchor_wp.get("wpbench_score") is not None else None)
        anchor_overall = anchor_wp.get("wpbench_score")
        material_shift = (anchor_overall is not None
                          and (INHERITED_FLOOR - anchor_overall) > SEED_NOISE_FLOOR)
        fresh_anchor = {
            "served_model_dir": "models/Qwen3.6-35B-A3B",
            "wpbench_overall": anchor_overall,
            "results_file": str(anchor_results_json) if anchor_results_json else None,
            "seed_noise_floor_reference": SEED_NOISE_FLOOR,
            "material_shift": material_shift,
            "justification": (
                f"raw new-base anchor {anchor_overall} vs inherited floor {INHERITED_FLOOR}: "
                + (f"gap {round(INHERITED_FLOOR - anchor_overall, 4)} EXCEEDS the seed-noise-floor "
                   f"reference {SEED_NOISE_FLOOR} -- material downward shift, fresh floor adopted."
                   if material_shift else
                   f"gap {round(INHERITED_FLOOR - anchor_overall, 4) if anchor_overall is not None else 'N/A'} "
                   f"does NOT exceed the seed-noise-floor reference {SEED_NOISE_FLOOR} -- no evidence "
                   f"of a material noise-band shift; the inherited floor stands and this miss is "
                   f"recorded as a valid, non-forced outcome.")
            ),
        }
        if material_shift:
            floor = anchor_overall
            floor_source = "fresh_new_base_anchor"
            pass_ = ci["ci_lower"] >= floor

    result = {
        "wpbench_overall": ci["point"],
        "wpbench_ci_lower": ci["ci_lower"],
        "wpbench_ci_upper": ci["ci_upper"],
        "floor": floor,
        "floor_source": floor_source,
        "pass": bool(pass_),
        "n_tests": ci["n_knowledge"] + ci["n_execution"],
        "n_knowledge": ci["n_knowledge"],
        "n_execution": ci["n_execution"],
        "seed": 1337,
        "max_tokens": 2048,
        "enable_thinking": False,
        "concurrency": 4,
        "temperature": 0.0,
        "request_timeout_s": 1800.0,
        "n_boot": ci["n_boot"],
        "alpha": ci["alpha"],
        "served_model_dir": merged_dir,
        "results_file": str(results_json),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    if fresh_anchor is not None:
        result["fresh_new_base_anchor"] = fresh_anchor

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[gen03] wrote {OUT_PATH}", flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VllmBootTimeout as e:
        print(f"HALT: vLLM boot timeout: {e}", file=sys.stderr)
        sys.exit(4)
