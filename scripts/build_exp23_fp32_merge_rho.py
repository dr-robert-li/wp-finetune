#!/usr/bin/env python
"""Phase 21 Diagnostic Experiment 3 -- fp32-accumulation merge fix, re-merge +
re-serve + re-score s1 judge rho (judge_attenuation_forensics.md Sec 5,
DIAGNOSTIC_SYNTHESIS.md #3).

Re-merges the SAME promoted s1 judge adapter (already on disk from Experiment
2/21-06 -- no re-download) via the fp32-upcast-adapter merge_adapter.py path
(scripts/merge_adapter.py's _fp32_upcast_adapter_copy /
_upcast_lora_layers_to_fp32), to a NEW versioned output dir (does not touch
or delete the original 21-06 canonical merged model, and naturally avoids
merge_adapter.py's adapter-content-hash idempotency short-circuit, which
would otherwise skip re-merging since the ADAPTER bytes are unchanged -- only
the merge CODE changed).

Re-serves via vLLM (same protocol as 21-06/build_judge03_merge_serve.py:
MAX_MODEL_LEN=16384, max_tokens=8192), re-captures the identical 121
wp_judge val prompts, and re-scores with the unmodified eval_relabel.py.

Decision rule: served rho recovers toward 0.8358 (the Tinker-capture anchor)
=> merge numerics were the (or a) cause, fp32 fix is ship-grade; rho stays at
~0.7872 (the ORIGINAL bf16-merge served figure) => engine numerics dominate,
the fp32 fix does not help.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "output" / "base21"
DIAG_DIR = OUT_DIR / "diagnostic"
CONFIG_PATH = "config/train_config_v4.yaml"
EXPECTED_MODULES_MANIFEST = OUT_DIR / "moe_merge_probe.json"
ADAPTER_DIR = str(OUT_DIR / "judge03_s1_adapter")
MERGED_DIR = "models/Qwen3.6-35B-A3B-judge-v4-s1-fp32merged"  # NEW versioned dir, original 21-06 merge untouched
DATASET = "data/reasoning_dataset/openai_val.jsonl"
DIFF_PROMPT = "<wp_judge> Evaluate this WordPress code:\n\n```php\nfunction f() { return 1; }\n```"
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
PORT = 8025
GPU_MEM_UTIL = 0.80
MAX_MODEL_LEN = 16384
MAX_TOKENS = 8192
BOOT_TIMEOUT_SEC = 1200

ORIGINAL_SERVED_RHO = 0.7872  # judge03_rho.json (bf16 merge, pre-fix)
CAPTURE_ANCHOR_RHO = 0.8358149892119933  # judge03_capture_rho.json (Tinker capture, target to recover toward)
RECOVERY_BAND = 0.02  # within this of the capture anchor => "recovered"


def _run_merge() -> dict:
    guard_receipt_path = OUT_DIR / "_exp3_fp32_merge_guard_result.json"
    cmd = [
        sys.executable, "scripts/merge_adapter.py",
        "--config-path", CONFIG_PATH,
        "--adapter-dir", ADAPTER_DIR,
        "--output-dir", MERGED_DIR,
        "--expected-modules-manifest", str(EXPECTED_MODULES_MANIFEST),
        "--guard-receipt-path", str(guard_receipt_path),
    ]
    print(f"[exp3-merge] running: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    print(r.stdout[-10000:])
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"merge_adapter.py (fp32 fix) exited {r.returncode}")
    print("[exp3-merge] subprocess exited 0", flush=True)
    return json.loads(guard_receipt_path.read_text())


def _dir_size_gib(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / (1024 ** 3), 2)


def _run_base_vs_merged_diff(merged_dir: str) -> bool:
    from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm

    def _serve(model_dir: str, container: str, allow_empty: bool) -> str:
        try:
            boot_vllm(model_dir, container, PORT, GPU_MEM_UTIL,
                      serve_script=SERVE_SCRIPT,
                      extra_env={"LANGUAGE_MODEL_ONLY": "1", "MAX_MODEL_LEN": str(MAX_MODEL_LEN)})
            served = wait_healthy(PORT, container, timeout=BOOT_TIMEOUT_SEC)
            out = generate(PORT, served,
                            [{"instruction": DIFF_PROMPT, "source_val_idx": "exp3_merge_diff"}],
                            max_tokens=256)
            text = (out[0] or "").strip()
            if not text and not allow_empty:
                raise RuntimeError(f"real-generation returned empty output for {container}")
            return text
        finally:
            stop_vllm(container)

    print("[exp3-serve] fp32-merged judge model ...", flush=True)
    merged_out = _serve(merged_dir, "exp3-fp32merged-diff", allow_empty=True)
    print(f"[exp3-serve] merged output: {merged_out[:200]!r}", flush=True)

    print("[exp3-serve] raw base model (for diff) ...", flush=True)
    base_out = _serve("models/Qwen3.6-35B-A3B", "exp3-base-diff", allow_empty=False)
    print(f"[exp3-serve] base output: {base_out[:200]!r}", flush=True)

    if not merged_out:
        return False
    return merged_out != base_out


def _capture_and_score(merged_dir: str) -> dict:
    from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm
    from scripts.sieve_capture_judge_http import capture as http_capture

    container = "exp3-fp32merged-eval"
    cap_path = OUT_DIR / "exp3_judge_capture_fp32merged_s1.jsonl"
    try:
        boot_vllm(merged_dir, container, PORT, GPU_MEM_UTIL,
                  serve_script=SERVE_SCRIPT,
                  extra_env={"LANGUAGE_MODEL_ONLY": "1", "MAX_MODEL_LEN": str(MAX_MODEL_LEN)})
        served = wait_healthy(PORT, container, timeout=BOOT_TIMEOUT_SEC)
        warm = generate(PORT, served,
                         [{"instruction": "Reply with exactly one word: OK", "source_val_idx": "warmup"}],
                         max_tokens=16)
        if not warm or not warm[0].strip():
            raise RuntimeError(f"real-generation warm-up returned empty output: {warm!r}")
        print(f"[warmup] real-generation OK (served_model={served!r}): {warm[0].strip()[:80]!r}", flush=True)

        cap_stats = http_capture(base_url=f"http://localhost:{PORT}/v1", model=served,
                                  dataset=DATASET, out=str(cap_path),
                                  max_tokens=MAX_TOKENS, temperature=0.0)
        print(f"[exp3-serve] capture stats: {cap_stats}", flush=True)
    finally:
        stop_vllm(container)

    r = subprocess.run([sys.executable, "scripts/relabel/eval_relabel.py", str(cap_path)],
                        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"eval_relabel failed on fp32-merged capture (exit {r.returncode})")
    summary = json.loads((OUT_DIR / "eval_summary.json").read_text())
    import re
    m = re.search(r"parse_fail=(\d+)", r.stdout)
    parse_fail = int(m.group(1)) if m else summary.get("parse_fail")

    return {
        "rho": summary["rho_new"],
        "ci_lower": summary["ci"][0],
        "ci_upper": summary["ci"][1],
        "n": summary["n"],
        "parse_fail": parse_fail,
        "max_tokens": MAX_TOKENS,
        "served_model_dir": merged_dir,
        "capture_path": str(cap_path),
    }


def _decision(rho: float) -> dict:
    delta_vs_original = round(rho - ORIGINAL_SERVED_RHO, 4)
    delta_vs_capture_anchor = round(rho - CAPTURE_ANCHOR_RHO, 4)
    recovered = abs(rho - CAPTURE_ANCHOR_RHO) <= RECOVERY_BAND
    if recovered:
        verdict = "merge numerics FIXED (ship-grade): served rho recovered toward the capture anchor"
    elif delta_vs_original > 0.01:
        verdict = "partial recovery: merge numerics are A contributor but do not fully explain the gap"
    else:
        verdict = "no material change: engine numerics dominate, fp32 merge fix does not help"
    return {
        "verdict": verdict,
        "rho_measured": rho,
        "original_served_rho_bf16_merge": ORIGINAL_SERVED_RHO,
        "capture_anchor_rho": CAPTURE_ANCHOR_RHO,
        "delta_vs_original_served": delta_vs_original,
        "delta_vs_capture_anchor": delta_vs_capture_anchor,
        "recovery_band": RECOVERY_BAND,
        "recovered": recovered,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAG_DIR / "exp3_fp32_merge_rho.json"
    t0 = time.time()

    guard = _run_merge()
    merged_target_module_count = guard["merged_target_module_count"]
    expected_target_module_count = guard["expected_target_module_count"]
    merge_ok = (merged_target_module_count == expected_target_module_count
                and merged_target_module_count > 0)

    print("[exp3] serving fp32-merged vs base for a real-generation diff ...", flush=True)
    base_vs_merged_differs = _run_base_vs_merged_diff(MERGED_DIR)
    if not (merge_ok and base_vs_merged_differs):
        result = {
            "experiment": "exp3_fp32_merge",
            "status": "blocked",
            "blocked_reason": f"merge guard failed: merge_ok={merge_ok} base_vs_merged_differs={base_vs_merged_differs}",
            "merge_ok": merge_ok,
            "merged_target_module_count": merged_target_module_count,
            "expected_target_module_count": expected_target_module_count,
            "base_vs_merged_differs": base_vs_merged_differs,
            "wall_clock_s": round(time.time() - t0, 1),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    vllm_served = _capture_and_score(MERGED_DIR)

    result = {
        "experiment": "exp3_fp32_merge",
        "status": "measured",
        "adapter_dir": ADAPTER_DIR,
        "merged_dir": MERGED_DIR,
        "merge_ok": merge_ok,
        "merged_target_module_count": merged_target_module_count,
        "expected_target_module_count": expected_target_module_count,
        "base_vs_merged_differs": base_vs_merged_differs,
        "merged_size_gib": _dir_size_gib(MERGED_DIR),
        "vllm_served_fp32_merged": vllm_served,
        "decision": _decision(vllm_served["rho"]),
        "max_model_len_served": MAX_MODEL_LEN,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[exp3] wrote {out_path}", flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
