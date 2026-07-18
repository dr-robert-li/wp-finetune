#!/usr/bin/env python
"""Phase 21 Diagnostic Experiment 1 -- bench a preserved gen sampler epoch
checkpoint on the full 344-test wp-bench suite (DIAGNOSTIC_SYNTHESIS.md #1,
gen_regression_forensics.md's ep3-overtraining hypothesis).

Downloads a NON-promoted epoch checkpoint from output/tinker/wp-gen-v4-
manifest.json (ep1 by default; ep2 optional), merges it via the exact same
routed-MoE-expert path as GEN-03 (21-05) -- which now includes Experiment 3's
fp32-accumulation fix (baked into merge_adapter.py, applies automatically) --
serves it, and runs the identical wp-bench harness/CI-aware bootstrap 21-05
used, so the result is directly comparable to:
  - ep3 (promoted, shipped): 0.372 overall, CI [0.2847, 0.4753]  (gen03_wpbench.json)
  - RAW new base (no adapter): 0.4897 overall  (gen03_wpbench.json fresh anchor)

Decision rule: ep1 >> ep3 (materially closes the gap to the raw-base anchor)
=> overtraining confirmed as a major contributor. ep1 ~= ep3 => data-shape
(training-target structure) dominates, consistent with gen_regression_
forensics.md's finding (92% of wp_gen targets are bare unwired fragments).

Usage:
    .venv-tinker/bin/python scripts/build_exp1_epoch_wpbench.py --epoch 1
    .venv-tinker/bin/python scripts/build_exp1_epoch_wpbench.py --epoch 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm, VllmBootTimeout  # noqa: E402
import scripts.run_eval_reasoning as rer  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "base21"
DIAG_DIR = OUT_DIR / "diagnostic"
MANIFEST_PATH = PROJECT_ROOT / "output" / "tinker" / "wp-gen-v4-manifest.json"
EXPECTED_MODULES_MANIFEST = OUT_DIR / "moe_merge_probe.json"
CONFIG_PATH = "config/train_config_v4.yaml"
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
PORT = 8024
GPU_MEM_UTIL = 0.80
DIFF_PROMPT = "Write a WordPress function that returns the current logged-in user's display name."
N_BOOT = 1000
ALPHA = 0.05
BOOTSTRAP_SEED = 1337

EP3_ANCHOR = 0.372  # gen03_wpbench.json (promoted, shipped)
RAW_BASE_ANCHOR = 0.4897  # gen03_wpbench.json fresh_new_base_anchor


def _adapter_dir(epoch: int) -> Path:
    return OUT_DIR / f"exp1_gen_ep{epoch}_adapter"


def _merged_dir(epoch: int) -> str:
    return f"models/Qwen3.6-35B-A3B-gen-v4-ep{epoch}-merged"


def _download_epoch_adapter(epoch: int) -> str:
    import tinker

    manifest = json.loads(MANIFEST_PATH.read_text())
    ckpt_name = f"wp-gen-v4-ep{epoch}"
    sampler_path = next(c["sampler_path"] for c in manifest["checkpoints"] if c["name"] == ckpt_name)
    print(f"[exp1] epoch {epoch} checkpoint: {ckpt_name} -> {sampler_path}", flush=True)

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    resp = rc.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
    url = getattr(resp, "url", None) or getattr(resp, "archive_url", None)
    if not url:
        raise RuntimeError(f"no archive URL in response: {resp!r}")

    adapter_dir = _adapter_dir(epoch)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tar_path = adapter_dir / "checkpoint.tar"
    print(f"[exp1] downloading archive -> {tar_path}", flush=True)
    r = subprocess.run(
        ["curl", "-L", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "10",
         "-o", str(tar_path), url],
        timeout=3600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl download failed (exit {r.returncode}) for {tar_path}")
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if name in ("adapter_config.json", "adapter_model.safetensors") and m.isfile():
                m.name = name
                tf.extract(m, adapter_dir)

    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / required).exists():
            raise RuntimeError(f"tinker archive missing {required} after extraction")

    return sampler_path


def _run_merge(epoch: int) -> dict:
    merged_dir = _merged_dir(epoch)
    guard_receipt_path = OUT_DIR / f"_exp1_ep{epoch}_merge_guard_result.json"
    cmd = [
        sys.executable, "scripts/merge_adapter.py",
        "--config-path", CONFIG_PATH,
        "--adapter-dir", str(_adapter_dir(epoch)),
        "--output-dir", merged_dir,
        "--expected-modules-manifest", str(EXPECTED_MODULES_MANIFEST),
        "--guard-receipt-path", str(guard_receipt_path),
    ]
    print(f"[exp1] merge: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    print(r.stdout[-10000:])
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"merge_adapter.py exited {r.returncode}")
    print("[exp1] merge subprocess exited 0", flush=True)
    return json.loads(guard_receipt_path.read_text())


def _dir_size_gib(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / (1024 ** 3), 2)


def _run_base_vs_merged_diff(merged_dir: str, epoch: int) -> bool:
    def _serve(model_dir: str, container: str, allow_empty: bool) -> str:
        try:
            boot_vllm(model_dir, container, PORT, GPU_MEM_UTIL,
                      serve_script=SERVE_SCRIPT, extra_env={"LANGUAGE_MODEL_ONLY": "1"})
            served = wait_healthy(PORT, container, timeout=1200)
            out = generate(PORT, served,
                            [{"instruction": DIFF_PROMPT, "source_val_idx": f"exp1_ep{epoch}_diff"}],
                            max_tokens=128)
            text = (out[0] or "").strip()
            if not text and not allow_empty:
                raise RuntimeError(f"real-generation returned empty output for {container}")
            return text
        finally:
            stop_vllm(container)

    print(f"[exp1] serving ep{epoch}-merged for diff ...", flush=True)
    merged_out = _serve(merged_dir, f"exp1-ep{epoch}-merged-diff", allow_empty=True)
    print(f"[exp1] merged output: {merged_out[:200]!r}", flush=True)

    print("[exp1] serving raw base for diff ...", flush=True)
    base_out = _serve("models/Qwen3.6-35B-A3B", f"exp1-ep{epoch}-base-diff", allow_empty=False)
    print(f"[exp1] base output: {base_out[:200]!r}", flush=True)

    if not merged_out:
        return False
    return merged_out != base_out


def _real_generation_warmup(served: str) -> None:
    warm = generate(PORT, served, [{"instruction": "Reply with exactly one word: OK", "source_val_idx": "warmup"}],
                     max_tokens=16)
    if not warm or not warm[0].strip():
        raise RuntimeError(f"Real-generation warm-up returned empty output: {warm!r}")
    print(f"[warmup] real-generation OK (served_model={served!r}): {warm[0].strip()[:80]!r}", file=sys.stderr)


def _run_wpbench_on(model_dir: str, container: str, tag: str) -> dict:
    t0 = time.time()
    try:
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


def _wp_bench_overall(knowledge_mean, correctness_mean, quality_mean) -> float:
    weights = {"knowledge": 0.3, "correctness": 0.4, "quality": 0.3}
    values = {"knowledge": knowledge_mean, "correctness": correctness_mean, "quality": quality_mean}
    active = {k: w for k, w in weights.items() if values[k] is not None}
    if not active:
        return 0.0
    total_weight = sum(active.values())
    total = sum(values[k] * w for k, w in active.items())
    return round(total / total_weight, 4)


def _bootstrap_ci_lower(results_json_path: Path, n_boot: int = N_BOOT, alpha: float = ALPHA) -> dict:
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

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_overall = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        k_resample = rng.choice(knowledge, size=knowledge.size, replace=True) if knowledge.size else knowledge
        c_resample = rng.choice(correctness, size=correctness.size, replace=True) if correctness.size else correctness
        q_resample = rng.choice(quality, size=quality.size, replace=True) if quality.size else quality
        boot_overall[i] = _wp_bench_overall(
            float(k_resample.mean()) if k_resample.size else None,
            float(c_resample.mean()) if c_resample.size else None,
            float(q_resample.mean()) if q_resample.size else None,
        )

    lo = float(np.percentile(boot_overall, 100 * alpha / 2))
    hi = float(np.percentile(boot_overall, 100 * (1 - alpha / 2)))
    return {
        "point": point, "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
        "n_knowledge": int(knowledge.size), "n_execution": int(correctness.size),
        "n_boot": n_boot, "alpha": alpha, "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _newest_results_json(tag: str) -> Path:
    cands = sorted((OUT_DIR / tag).glob("wp_bench_results_*.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise RuntimeError(f"no wp_bench_results_*.json found under {OUT_DIR / tag}")
    return cands[-1]


def _decision(epoch: int, overall: float) -> dict:
    delta_vs_ep3 = round(overall - EP3_ANCHOR, 4)
    delta_vs_raw = round(overall - RAW_BASE_ANCHOR, 4)
    # "materially closes the gap": recovers at least half the ep3-vs-raw-base deficit
    ep3_to_raw_gap = RAW_BASE_ANCHOR - EP3_ANCHOR
    recovered_fraction = (overall - EP3_ANCHOR) / ep3_to_raw_gap if ep3_to_raw_gap else 0.0
    if recovered_fraction >= 0.5:
        verdict = "overtraining CONFIRMED as a major contributor (ep{} materially closes the ep3-vs-raw-base gap)".format(epoch)
    elif recovered_fraction >= 0.15:
        verdict = "overtraining is A contributor but does not fully explain the regression"
    else:
        verdict = "data-shape dominates (ep{} ~= ep3, consistent with gen_regression_forensics.md)".format(epoch)
    return {
        "verdict": verdict,
        "epoch": epoch,
        "overall_measured": overall,
        "ep3_anchor": EP3_ANCHOR,
        "raw_base_anchor": RAW_BASE_ANCHOR,
        "delta_vs_ep3": delta_vs_ep3,
        "delta_vs_raw_base": delta_vs_raw,
        "recovered_fraction_of_ep3_to_raw_gap": round(recovered_fraction, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, required=True, choices=[1, 2])
    args = ap.parse_args()
    epoch = args.epoch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAG_DIR / f"exp1_ep{epoch}_wpbench.json"
    t0 = time.time()

    sampler_path = _download_epoch_adapter(epoch)
    guard = _run_merge(epoch)
    merged_dir = _merged_dir(epoch)
    merged_target_module_count = guard["merged_target_module_count"]
    expected_target_module_count = guard["expected_target_module_count"]
    merge_ok = (merged_target_module_count == expected_target_module_count
                and merged_target_module_count > 0)

    base_vs_merged_differs = _run_base_vs_merged_diff(merged_dir, epoch)
    if not (merge_ok and base_vs_merged_differs):
        result = {
            "experiment": "exp1_epoch_wpbench", "epoch": epoch, "status": "blocked",
            "blocked_reason": f"merge guard failed: merge_ok={merge_ok} base_vs_merged_differs={base_vs_merged_differs}",
            "merged_dir": merged_dir, "merged_target_module_count": merged_target_module_count,
            "expected_target_module_count": expected_target_module_count,
            "wall_clock_s": round(time.time() - t0, 1),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    wp = _run_wpbench_on(merged_dir, f"exp1-ep{epoch}-wpbench-vllm", f"exp1_ep{epoch}_full")
    if not wp.get("ran") or wp.get("wpbench_score") is None:
        result = {
            "experiment": "exp1_epoch_wpbench", "epoch": epoch, "status": "error",
            "error": f"wp-bench failed: {json.dumps(wp)[:3000]}",
            "wall_clock_s": round(time.time() - t0, 1),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 3

    results_json = _newest_results_json(f"exp1_ep{epoch}_full")
    ci = _bootstrap_ci_lower(results_json)

    result = {
        "experiment": "exp1_epoch_wpbench",
        "status": "measured",
        "epoch": epoch,
        "promoted_sampler_path": sampler_path,
        "merged_dir": merged_dir,
        "merge_ok": merge_ok,
        "merged_target_module_count": merged_target_module_count,
        "expected_target_module_count": expected_target_module_count,
        "base_vs_merged_differs": base_vs_merged_differs,
        "merged_size_gib": _dir_size_gib(merged_dir),
        "wpbench_overall": ci["point"],
        "wpbench_ci_lower": ci["ci_lower"],
        "wpbench_ci_upper": ci["ci_upper"],
        "n_tests": ci["n_knowledge"] + ci["n_execution"],
        "n_knowledge": ci["n_knowledge"],
        "n_execution": ci["n_execution"],
        "seed": 1337,
        "max_tokens": 2048,
        "enable_thinking": False,
        "concurrency": 4,
        "temperature": 0.0,
        "n_boot": ci["n_boot"],
        "alpha": ci["alpha"],
        "bootstrap_seed": ci["bootstrap_seed"],
        "results_file": str(results_json),
        "decision": _decision(epoch, ci["point"]),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[exp1] wrote {out_path}", flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VllmBootTimeout as e:
        print(f"HALT: vLLM boot timeout: {e}", file=sys.stderr)
        sys.exit(4)
