#!/usr/bin/env python
"""SIEVE-04 k-sweep driver (plan 11-04 Task 2).

For k in {13, 32, 64, "full"}: build the inference-time expert keep-mask
(scripts.sieve_expert_mask_inference), serve gen + 3 judge seeds SEQUENTIALLY
(GB10 memory wall -- never two 30B instances resident at once), measure
wp-bench (gen axis, full 344-test suite) and judge ensemble rho (3-seed
median vs val_labels_v1, judge axis), and write output/sieve/k_sweep_results.json.

Arm order: full first (so the pre-registered sanity bounds -- judge_ensemble_rho
>= 0.822, wp_bench >= 0.4416 -- gate the WHOLE sweep before burning GPU time on
masked arms), then 64, 32, 13 (decreasing k).

No training, no gradients: masking is a runtime router-logit patch
(scripts/_sieve_vllm_patch/sitecustomize.py), applied only while a given
container is up.

Usage (run as ONE backgrounded driver, per plan instructions):
    nohup .venv-tinker/bin/python -m scripts.sieve_ksweep_run \
        > logs/sieve/ksweep_driver.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm, VllmBootTimeout  # noqa: E402
from scripts.sieve_expert_mask_inference import (  # noqa: E402
    build_ksweep_mask, build_profile_counts, load_protected_mask,
)

GEN_MODEL = "models/qwen3-30b-wp-30_70-reasoning-merged-v4"
GEN_ROUTING = ["output/profiling/reasoning-merged-v4/routing_report.jsonl"]
PROTECTED_MASK = "output/profiling/reasoning-merged-v4/protected_expert_mask.npy"

JUDGE_SEEDS = {
    "s0": "models/_staging/qwen3-30b-wp-v1.3-s0-merged",
    "s1": "models/_staging/qwen3-30b-wp-v1.3-merged",
    "s2": "models/_staging/qwen3-30b-wp-v1.3-s2-merged",
}
JUDGE_ROUTING = [
    "output/sieve/judge-s0/routing_report.jsonl",
    "output/sieve/judge-s1/routing_report.jsonl",
    "output/sieve/judge-s2/routing_report.jsonl",
]  # sieve_profile_mode=shared (11-03): ONE profile (summed) covers all 3 seeds

VAL_DATASET = "data/reasoning_dataset/openai_val.jsonl"
VAL_LABELS = "output/relabel/val_labels_v1.json"

PORT = 8021  # matches run_eval_reasoning.PORT default (gen axis reuses that module)
GPU_MEM_UTIL = 0.55
KS = [64, 32, 13]  # + "full", full runs first (see main())

MASK_DIR = PROJECT_ROOT / "output/sieve/masks"
OUT_ROOT = PROJECT_ROOT / "output/sieve/ksweep"
RESULTS_PATH = PROJECT_ROOT / "output/sieve/k_sweep_results.json"

# Pre-registered sanity bounds (11-04-PLAN.md acceptance criteria). Full-arm
# failure -> HALT, harness misconfiguration not a masking result.
FULL_JUDGE_RHO_FLOOR = 0.822
FULL_WPBENCH_FLOOR = 0.4416


def _set_mask_env(mask_path: Path | None) -> None:
    if mask_path is None:
        os.environ.pop("SIEVE_MASK_NPY", None)
    else:
        os.environ["SIEVE_MASK_NPY"] = str(mask_path)


def build_masks_for_k(k: int | str) -> tuple[Path | None, Path | None, dict]:
    """Returns (gen_mask_path, judge_mask_path, meta). None, None for k="full"."""
    if k == "full":
        return None, None, {"gen_kept_experts_per_layer": None,
                             "judge_kept_experts_per_layer": None,
                             "protected_retained": True}
    protected = load_protected_mask(PROJECT_ROOT / PROTECTED_MASK)

    gen_counts = build_profile_counts([PROJECT_ROOT / p for p in GEN_ROUTING])
    gen_keep = build_ksweep_mask(gen_counts, protected, k)
    gen_path = MASK_DIR / f"gen_k{k}.npy"
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(gen_path, gen_keep)

    judge_counts = build_profile_counts([PROJECT_ROOT / p for p in JUDGE_ROUTING])
    judge_keep = build_ksweep_mask(judge_counts, protected, k)
    judge_path = MASK_DIR / f"judge_shared_k{k}.npy"
    np.save(judge_path, judge_keep)

    protected_retained = bool(np.all((~protected) | gen_keep) and np.all((~protected) | judge_keep))
    meta = {
        "gen_kept_experts_per_layer": gen_keep.sum(axis=1).tolist(),
        "judge_kept_experts_per_layer": judge_keep.sum(axis=1).tolist(),
        "protected_retained": protected_retained,
    }
    return gen_path, judge_path, meta


def _reset_wpbench_grader() -> None:
    """Stop+remove any wp-env-runtime-* grader containers before a wp-bench run.

    Discovered live during this plan's execution (not a masking bug): wp-bench's
    docker-based PHP-execution grader (config/wp-bench.yaml grader.wp_env_dir)
    REUSES its WordPress+MySQL containers across separate invocations rather
    than recreating them. Across sequential k-sweep arms this let WordPress-DB
    state accumulate from earlier test executions, silently degrading the
    "correctness" (code-execution) sub-score run-to-run while "knowledge"
    (pure text matching, no execution) stayed bit-identical -- confirmed via
    3 repeat invocations: 0.4603 (fresh containers) -> 0.4365 (stale, twice,
    byte-identical per-test) -> 0.4603 (fresh again after this reset). Not
    vLLM/model nondeterminism (generation was byte-identical across all
    invocations); a stale wp-bench-owned Docker fixture. Reset before every
    gen arm so each k gets a fair, reproducible correctness measurement.
    """
    import subprocess
    names = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"],
                            capture_output=True, text=True).stdout.splitlines()
    stale = [n for n in names if n.startswith("wp-env-runtime-")]
    if stale:
        print(f"[wpbench-reset] removing stale grader containers: {stale}", flush=True)
        subprocess.run(["docker", "rm", "-f", *stale], stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT, check=False)


def run_gen_arm(k: int | str, gen_mask_path: Path | None) -> dict:
    """Boot gen model (masked or full), run REVL-04-style wp-bench, stop."""
    from scripts.run_eval_reasoning import _wpbench_with_boot

    _reset_wpbench_grader()
    tag = f"gen_k{k}"
    _set_mask_env(gen_mask_path)
    try:
        res = _wpbench_with_boot(GEN_MODEL, f"sieve-gen-k{k}", tag, GPU_MEM_UTIL, OUT_ROOT)
    finally:
        _set_mask_env(None)
    return res


def capture_judge_seed(k: int | str, seed: str, judge_mask_path: Path | None) -> Path:
    """Boot one judge seed (masked or full), capture val-set judge responses, stop."""
    from scripts.sieve_capture_judge_http import capture

    name = f"sieve-judge-{seed}-k{k}"
    out = OUT_ROOT / f"judge_k{k}" / seed / "judge_responses.jsonl"
    _set_mask_env(judge_mask_path)
    try:
        boot_vllm(JUDGE_SEEDS[seed], name, PORT, GPU_MEM_UTIL)
        wait_healthy(PORT, name)
        capture(base_url=f"http://localhost:{PORT}/v1", model=None,
                dataset=VAL_DATASET, out=str(out))
    finally:
        _set_mask_env(None)
        stop_vllm(name)
    return out


def score_judge_ensemble(seed_captures: dict[str, Path]) -> dict:
    """3-seed median ensemble rho + single-s1 rho vs val_labels_v1 (n=121 val set)."""
    from eval.eval_judge import _derive_prose_overall
    from eval.output_parsers import load_dim_map, parse_judge_scores
    from scipy.stats import spearmanr

    dm = load_dim_map()
    dw = {kk: v for kk, v in dm["dimension_weights"].items() if not kk.startswith("_")}
    rows = [json.loads(line) for line in open(PROJECT_ROOT / VAL_DATASET) if line.strip()]
    wj_rows = [i for i, r in enumerate(rows) if next(
        (m["content"] for m in r["messages"] if m["role"] == "user"), ""
    ).startswith("<wp_judge>")]
    labels = {kk: v for kk, v in json.load(open(PROJECT_ROOT / VAL_LABELS)).items()
              if kk.startswith("val:")}

    def load_capture(path: Path) -> dict:
        scores = {}
        for line in open(path):
            r = json.loads(line)
            if "index" not in r:
                continue
            parsed = parse_judge_scores(r["response"], "auto")
            if not parsed or not parsed.get("dimension_scores"):
                continue
            o = (float(parsed["overall"]) if "overall" in parsed
                 else _derive_prose_overall(parsed["dimension_scores"], dw))
            scores[f"val:{wj_rows[r['index']]}"] = o
        return scores

    per_seed = {seed: load_capture(p) for seed, p in seed_captures.items()}

    ensemble = {}
    for key in labels:
        vals = [per_seed[s][key] for s in per_seed if key in per_seed[s]]
        if vals:
            ensemble[key] = float(np.median(vals))
    common = sorted(set(ensemble) & set(labels))
    ens_rho = (spearmanr([ensemble[kk] for kk in common],
                         [labels[kk] for kk in common]).statistic
               if len(common) > 2 else None)

    single_s1_rho = None
    if "s1" in per_seed:
        common_s1 = sorted(set(per_seed["s1"]) & set(labels))
        if len(common_s1) > 2:
            single_s1_rho = spearmanr([per_seed["s1"][kk] for kk in common_s1],
                                      [labels[kk] for kk in common_s1]).statistic

    return {"judge_ensemble_rho": ens_rho, "judge_single_s1_rho": single_s1_rho,
            "n_scored": len(common), "n_per_seed": {s: len(v) for s, v in per_seed.items()}}


def run_one_arm(k: int | str) -> dict:
    print(f"=== k={k}: building masks ===", flush=True)
    gen_mask_path, judge_mask_path, mask_meta = build_masks_for_k(k)

    print(f"=== k={k}: gen wp-bench (sequential serve) ===", flush=True)
    gen_res = run_gen_arm(k, gen_mask_path)

    print(f"=== k={k}: judge captures (3 seeds sequential serve) ===", flush=True)
    seed_captures = {}
    for seed in ("s0", "s1", "s2"):
        seed_captures[seed] = capture_judge_seed(k, seed, judge_mask_path)

    judge_res = score_judge_ensemble(seed_captures)

    arm = {
        "k": str(k),
        "wp_bench": gen_res.get("wpbench_score"),
        "wp_bench_detail": {kk: gen_res.get(kk) for kk in ("scores", "ran", "error")},
        "judge_ensemble_rho": judge_res["judge_ensemble_rho"],
        "judge_single_s1_rho": judge_res["judge_single_s1_rho"],
        "judge_n_scored": judge_res["n_scored"],
        "judge_n_per_seed": judge_res["n_per_seed"],
        "sieve_profile_mode": "shared",
        "masked_seeds": [] if k == "full" else ["s0", "s1", "s2"],
        **mask_meta,
    }
    return arm


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    order = ["full", *KS]
    sweep = []
    halted = False
    for k in order:
        t0 = time.time()
        arm = run_one_arm(k)
        arm["duration_sec"] = round(time.time() - t0, 1)
        sweep.append(arm)

        RESULTS_PATH.write_text(json.dumps({"sweep": sweep, "halted": halted}, indent=2))
        print(f"=== k={k} DONE: wp_bench={arm['wp_bench']} "
              f"judge_ensemble_rho={arm['judge_ensemble_rho']} "
              f"({arm['duration_sec']}s) ===", flush=True)

        if k == "full":
            wp = arm["wp_bench"]
            rho = arm["judge_ensemble_rho"]
            if wp is None or rho is None or wp < FULL_WPBENCH_FLOOR or rho < FULL_JUDGE_RHO_FLOOR:
                halted = True
                RESULTS_PATH.write_text(json.dumps({"sweep": sweep, "halted": True,
                    "halt_reason": (f"full arm sanity bounds failed: "
                                    f"wp_bench={wp} (floor {FULL_WPBENCH_FLOOR}), "
                                    f"judge_ensemble_rho={rho} (floor {FULL_JUDGE_RHO_FLOOR})")},
                    indent=2))
                print(f"HALT: full-arm sanity bounds failed (wp_bench={wp}, rho={rho}). "
                      "Aborting k-sweep -- harness misconfiguration, not a masking result.",
                      flush=True)
                return 1
    print("=== k-sweep COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
