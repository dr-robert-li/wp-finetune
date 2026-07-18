#!/usr/bin/env python
"""Phase 21 Diagnostic Experiment 4 -- bench the REBUILT-MIX gen checkpoint
(wp-gen-v4b, trained on data/reasoning_dataset/openai_train_v4_rebuilt.jsonl)
on the full 344-test wp-bench suite.

Reuses build_exp1_epoch_wpbench.py's merge/serve/bench/CI machinery verbatim
(imported, not copied) -- only the checkpoint source (v4b manifest promoted
ckpt), directory names, and the decision anchors differ. Directly comparable to:
  - ep3 (old mix, promoted, shipped): 0.372   (gen03_wpbench.json)
  - ep1 (old mix, exp1):              0.4381  (exp1_ep1_wpbench.json)
  - RAW new base (no adapter):        0.4897  (gen03_wpbench.json fresh anchor)

Decision rule: v4b > raw base => rebuilt mix fixed the regression AND adds value.
v4b ~= raw => regression fixed (no harm), mix adds no measurable gen skill.
v4b < raw but > ep3/ep1 => partial fix. v4b <= ep3 => rebuild failed.

Usage:
    .venv-tinker/bin/python scripts/build_exp4_wpbench.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse exp1's generic machinery (serve/diff/bench/CI) -- the lazy path.
from scripts.build_exp1_epoch_wpbench import (  # noqa: E402
    CONFIG_PATH, EXPECTED_MODULES_MANIFEST, OUT_DIR, DIAG_DIR,
    _bootstrap_ci_lower, _dir_size_gib, _newest_results_json,
    _run_base_vs_merged_diff, _run_wpbench_on,
)
from scripts._p0_vllm_smoke_serve import VllmBootTimeout  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "output" / "tinker" / "wp-gen-v4b-manifest.json"
ADAPTER_DIR = OUT_DIR / "exp4_gen_v4b_adapter"
MERGED_DIR = "models/Qwen3.6-35B-A3B-gen-v4b-merged"
OUT_PATH = DIAG_DIR / "exp4_bench.json"
BENCH_TAG = "exp4_v4b_full"

EP3_ANCHOR = 0.372     # old mix, 3 epochs (gen03_wpbench.json)
EP1_ANCHOR = 0.4381    # old mix, 1 epoch (exp1_ep1_wpbench.json)
RAW_BASE_ANCHOR = 0.4897  # untrained new base (gen03_wpbench.json fresh anchor)


def _download_promoted_adapter() -> str:
    import tinker

    manifest = json.loads(MANIFEST_PATH.read_text())
    promoted = manifest["promoted"]
    sampler_path = next(c["sampler_path"] for c in manifest["checkpoints"] if c["name"] == promoted)
    print(f"[exp4] promoted checkpoint: {promoted} -> {sampler_path}", flush=True)

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    resp = rc.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
    url = getattr(resp, "url", None) or getattr(resp, "archive_url", None)
    if not url:
        raise RuntimeError(f"no archive URL in response: {resp!r}")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = ADAPTER_DIR / "checkpoint.tar"
    print(f"[exp4] downloading archive -> {tar_path}", flush=True)
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
                tf.extract(m, ADAPTER_DIR)

    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (ADAPTER_DIR / required).exists():
            raise RuntimeError(f"tinker archive missing {required} after extraction")

    return sampler_path


def _run_merge() -> dict:
    guard_receipt_path = OUT_DIR / "_exp4_v4b_merge_guard_result.json"
    cmd = [
        sys.executable, "scripts/merge_adapter.py",
        "--config-path", CONFIG_PATH,
        "--adapter-dir", str(ADAPTER_DIR),
        "--output-dir", MERGED_DIR,
        "--expected-modules-manifest", str(EXPECTED_MODULES_MANIFEST),
        "--guard-receipt-path", str(guard_receipt_path),
    ]
    print(f"[exp4] merge: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    print(r.stdout[-10000:])
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"merge_adapter.py exited {r.returncode}")
    print("[exp4] merge subprocess exited 0", flush=True)
    return json.loads(guard_receipt_path.read_text())


def _decision(overall: float) -> dict:
    ep3_to_raw_gap = RAW_BASE_ANCHOR - EP3_ANCHOR
    recovered_fraction = (overall - EP3_ANCHOR) / ep3_to_raw_gap if ep3_to_raw_gap else 0.0
    if overall > RAW_BASE_ANCHOR:
        verdict = "rebuilt mix FIXED the regression and ADDS gen skill above the raw base"
    elif overall >= RAW_BASE_ANCHOR - 0.02:
        verdict = "rebuilt mix FIXED the regression (parity with raw base within noise); no measurable net gen gain"
    elif overall > EP1_ANCHOR:
        verdict = "PARTIAL fix: beats both old-mix checkpoints (ep1/ep3) but still below the raw base"
    elif overall > EP3_ANCHOR:
        verdict = "WEAK partial fix: beats shipped ep3 but not old-mix ep1 -- mix rebuild helped less than simply stopping earlier"
    else:
        verdict = "rebuild FAILED: at or below the old-mix ep3 result"
    return {
        "verdict": verdict,
        "overall_measured": overall,
        "ep3_old_mix_anchor": EP3_ANCHOR,
        "ep1_old_mix_anchor": EP1_ANCHOR,
        "raw_base_anchor": RAW_BASE_ANCHOR,
        "delta_vs_ep3": round(overall - EP3_ANCHOR, 4),
        "delta_vs_ep1": round(overall - EP1_ANCHOR, 4),
        "delta_vs_raw_base": round(overall - RAW_BASE_ANCHOR, 4),
        "recovered_fraction_of_ep3_to_raw_gap": round(recovered_fraction, 3),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sampler_path = _download_promoted_adapter()
    guard = _run_merge()
    merged_target_module_count = guard["merged_target_module_count"]
    expected_target_module_count = guard["expected_target_module_count"]
    merge_ok = (merged_target_module_count == expected_target_module_count
                and merged_target_module_count > 0)

    # epoch arg only tags container names in exp1's helper; 4 = exp4 marker
    base_vs_merged_differs = _run_base_vs_merged_diff(MERGED_DIR, epoch=4)
    if not (merge_ok and base_vs_merged_differs):
        result = {
            "experiment": "exp4_v4b_wpbench", "status": "blocked",
            "blocked_reason": f"merge guard failed: merge_ok={merge_ok} base_vs_merged_differs={base_vs_merged_differs}",
            "merged_dir": MERGED_DIR, "merged_target_module_count": merged_target_module_count,
            "expected_target_module_count": expected_target_module_count,
            "wall_clock_s": round(time.time() - t0, 1),
        }
        OUT_PATH.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    wp = _run_wpbench_on(MERGED_DIR, "exp4-v4b-wpbench-vllm", BENCH_TAG)
    if not wp.get("ran") or wp.get("wpbench_score") is None:
        result = {
            "experiment": "exp4_v4b_wpbench", "status": "error",
            "error": f"wp-bench failed: {json.dumps(wp)[:3000]}",
            "wall_clock_s": round(time.time() - t0, 1),
        }
        OUT_PATH.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 3

    results_json = _newest_results_json(BENCH_TAG)
    ci = _bootstrap_ci_lower(results_json)

    result = {
        "experiment": "exp4_v4b_wpbench",
        "status": "measured",
        "train_mix": "data/reasoning_dataset/openai_train_v4_rebuilt.jsonl (exp4_mix_provenance.json)",
        "promoted_sampler_path": sampler_path,
        "merged_dir": MERGED_DIR,
        "merge_ok": merge_ok,
        "merged_target_module_count": merged_target_module_count,
        "expected_target_module_count": expected_target_module_count,
        "base_vs_merged_differs": base_vs_merged_differs,
        "merged_size_gib": _dir_size_gib(MERGED_DIR),
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
        "decision": _decision(ci["point"]),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"[exp4] wrote {OUT_PATH}", flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VllmBootTimeout as e:
        print(f"HALT: vLLM boot timeout: {e}", file=sys.stderr)
        sys.exit(4)
