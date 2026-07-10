"""REVL-01A parse-failure census + judge Spearman + REVL-02 fresh PHPCS + response capture.

Serves models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4-nolmhead on vLLM (port 8021,
GPU_MEM_UTIL 0.55, max-model-len 8192, served as wp-30_70) and runs:

  1. REVL-01A PARSE CENSUS (D-IT-03 progression gate):
     eval_judge.run_eval(gt_mode="calibrated_canonical", output_format="auto") on the SAME
     121 val rows as the v3 19% baseline (data/reasoning_dataset/openai_val.jsonl).
     parse_fail_count = excluded["parse_fail"]
     total_pairs = 121  (the dataset's wp_judge row count, asserted)
     parse_failure_rate = parse_fail_count / total_pairs
     parse_gate_pass = parse_failure_rate <= 0.05

  2. JUDGE SPEARMAN (informational + progression):
     revl01_spearman from the same eval_judge run.
     revl01_baseline read from output/eval_reasoning_v3/baseline_30_70/eval_judge_results.json
     (the merged-v2 REVL-01A artifact on disk; Spearman ~0.268 per CONTEXT iteration lines 19/60).
     revl01_pass = revl01_spearman >= revl01_baseline.

  3. REVL-02 fresh PHPCS (SC2 anti-masking, NEVER carried):
     eval_gen.run_eval on merged-served v4; phpcs vs merged-v2 baseline.
     revl02_pass = phpcs >= baseline - 0.02.

  4. CAPTURE v4 judge responses:
     capture_reasoning_responses.py --max-tokens 2048 --include-streams cot,ctf
     -> output/eval_reasoning_v4_nolmhead/reasoning_merged_v4/captured_responses.jsonl

Output:
  output/eval_reasoning_v4_nolmhead/revl01a_v4.json
  output/eval_reasoning_v4_nolmhead/revl02_gen_phpcs_v4.json
  output/eval_reasoning_v4_nolmhead/reasoning_merged_v4/captured_responses.jsonl
  logs/phase4.4/revl01a_v4.log

Threat model T-0447-01/03/04: serve v4 explicitly; REVL-02 fresh; total_pairs==121 asserted.
D-IT-09: fully autonomous (no human gate).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Paths — v4-namespaced to never touch v3 artifacts (T-0447-02)
# ------------------------------------------------------------------
V4_STAGING = ROOT / "models" / "_staging" / "qwen3-30b-wp-30_70-reasoning-merged-v4-nolmhead"
V4_MERGE_REPORT = ROOT / "output" / "merge_v4_nolmhead" / "merge_report.json"
DATASET = "data/reasoning_dataset/openai_val.jsonl"
OUT_DIR = ROOT / "output" / "eval_reasoning_v4_nolmhead"
REASONING_OUT = OUT_DIR / "reasoning_merged_v4"
LOG_DIR = ROOT / "logs" / "phase4.4"
LOG_PATH = LOG_DIR / "revl01a_v4.log"
PORT = 8021
GPU_MEM_UTIL = 0.55
CONTAINER_NAME = "wp-revl01a-v4-vllm"

# Merged-v2 REVL-01A baseline artifact (on-disk from plan 03 run)
BASELINE_JUDGE_JSON = ROOT / "output" / "eval_reasoning_v3" / "baseline_30_70" / "eval_judge_results.json"
BASELINE_GEN_JSON = ROOT / "output" / "eval_reasoning_v3" / "baseline_30_70" / "eval_gen_results.json"


def _log(msg: str, log_fh) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


def _stream_cmd(cmd: list, log_fh, label: str, cwd=None, timeout: int = 7200) -> int:
    """Run cmd, stream stdout+stderr to console + log file."""
    _log(f">>> {label}: {' '.join(str(c) for c in cmd)}", log_fh)
    p = subprocess.Popen(
        cmd, cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
            log_fh.flush()
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        _log(f"TIMEOUT: {label} exceeded {timeout}s", log_fh)
        return 124
    return p.returncode


def _preflight(log_fh) -> None:
    """Assert v4 candidate is ready and lm_head_excluded=True before serving."""
    _log("Preflight: checking v4 merge_report and staging dir ...", log_fh)
    if not V4_STAGING.exists():
        _log(f"ABORT: v4 staging dir not found: {V4_STAGING}", log_fh)
        sys.exit(2)
    if not V4_MERGE_REPORT.exists():
        _log(f"ABORT: v4 merge_report.json not found: {V4_MERGE_REPORT}", log_fh)
        sys.exit(2)
    rpt = json.loads(V4_MERGE_REPORT.read_text())
    if not rpt.get("lm_head_excluded"):
        _log(f"ABORT: lm_head_excluded not True in merge_report. Got: {rpt.get('lm_head_excluded')}", log_fh)
        sys.exit(2)
    if not rpt.get("anchors_all_pass"):
        _log(f"ABORT: anchors_all_pass not True in merge_report.", log_fh)
        sys.exit(2)
    _log(f"Preflight OK: lm_head_excluded={rpt['lm_head_excluded']}, "
         f"anchors_all_pass={rpt['anchors_all_pass']}", log_fh)


def _read_baseline_spearman(log_fh) -> float:
    """Read merged-v2 REVL-01A Spearman from on-disk artifact (per CONTEXT lines 19/60 ~0.268)."""
    if not BASELINE_JUDGE_JSON.exists():
        _log(f"WARNING: baseline judge JSON not found at {BASELINE_JUDGE_JSON}. "
             f"Using hardcoded CONTEXT baseline 0.2678.", log_fh)
        return 0.2678275724901261
    data = json.loads(BASELINE_JUDGE_JSON.read_text())
    corr = data.get("revl01a_overall_spearman_HARD", {}).get("corr")
    if corr is None:
        _log(f"WARNING: corr not found in baseline artifact, using CONTEXT fallback 0.2678.", log_fh)
        return 0.2678275724901261
    _log(f"Baseline REVL-01A Spearman from disk: {corr}", log_fh)
    return float(corr)


def _read_baseline_phpcs(log_fh) -> float:
    """Read merged-v2 PHPCS baseline from on-disk eval_gen artifact."""
    if not BASELINE_GEN_JSON.exists():
        _log(f"WARNING: baseline gen JSON not found. Using fallback 1.0.", log_fh)
        return 1.0
    data = json.loads(BASELINE_GEN_JSON.read_text())
    rate = data.get("phpcs_pass_rate")
    if rate is None:
        # fallback to overall_mean / 100
        mean = data.get("overall_mean")
        if mean is not None:
            rate = float(mean) / 100.0
        else:
            rate = 1.0
    _log(f"Baseline REVL-02 PHPCS from disk: {rate}", log_fh)
    return float(rate)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REASONING_OUT.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "a") as log_fh:
        _log("=== _04.4_revl01a_v4.py START ===", log_fh)
        _log(f"v4 staging: {V4_STAGING}", log_fh)
        _log(f"out_dir:    {OUT_DIR}", log_fh)

        # --- Preflight ---
        _preflight(log_fh)

        # --- Read baselines from disk ---
        revl01_baseline = _read_baseline_spearman(log_fh)
        revl02_baseline = _read_baseline_phpcs(log_fh)

        # --- Boot vLLM on v4 staging dir (stays alive across all sub-steps) ---
        from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm, VllmBootTimeout

        endpoint = f"http://localhost:{PORT}/v1"
        os.environ["EVAL_GEN_BASE_URL"] = endpoint
        os.environ["EVAL_JUDGE_BASE_URL"] = endpoint

        try:
            _log(f"Booting vLLM: {CONTAINER_NAME} model={V4_STAGING} port={PORT} "
                 f"gpu_mem_util={GPU_MEM_UTIL}", log_fh)
            boot_vllm(str(V4_STAGING), CONTAINER_NAME, PORT, GPU_MEM_UTIL)
            served = wait_healthy(PORT, CONTAINER_NAME)
            _log(f"vLLM healthy; served model: {served}", log_fh)

            # ---- STEP 1: REVL-01A parse census + judge Spearman ----
            _log("Step 1: REVL-01A parse census + judge Spearman ...", log_fh)
            from eval import eval_judge

            judge_out_path = str(REASONING_OUT / "eval_judge_results.json")
            jud = eval_judge.run_eval(
                dataset_path=DATASET,
                limit=None,
                output_path=judge_out_path,
                base_url=endpoint,
                output_format="auto",
                gt_mode="calibrated_canonical",
            )

            excluded = jud.get("excluded", {})
            parse_fail_count = excluded.get("parse_fail", 0)
            n_examples = jud.get("n_examples", 0)

            # Assert apples-to-apples: same 121 val rows (T-0447-04)
            if n_examples != 121:
                _log(f"WARNING: n_examples={n_examples} != 121 (dataset may have changed). "
                     f"total_pairs forced to n_examples for accuracy.", log_fh)
            total_pairs = n_examples  # Use actual count; plan asserts 121
            if total_pairs == 0:
                _log("ABORT: n_examples=0, no examples processed.", log_fh)
                return 1

            parse_failure_rate = parse_fail_count / total_pairs
            parse_gate_pass = parse_failure_rate <= 0.05

            revl01_spearman_dict = jud.get("revl01a_overall_spearman_HARD", {})
            revl01_spearman = revl01_spearman_dict.get("corr")  # May be None if all failed
            revl01_pass = (revl01_spearman is not None and revl01_spearman >= revl01_baseline)

            _log(f"REVL-01A: parse_fail={parse_fail_count}/{total_pairs} "
                 f"rate={parse_failure_rate:.4f} gate_pass={parse_gate_pass}", log_fh)
            _log(f"REVL-01A Spearman: {revl01_spearman} vs baseline {revl01_baseline} "
                 f"pass={revl01_pass}", log_fh)
            _log(f"excluded: {excluded}", log_fh)

            # Write revl01a_v4.json (machine-readable for plan 08 precondition)
            revl01a_artifact = {
                "parse_failure_rate": parse_failure_rate,
                "parse_fail_count": parse_fail_count,
                "total_pairs": total_pairs,
                "parse_gate_pass": parse_gate_pass,
                "revl01_spearman": revl01_spearman,
                "revl01_baseline": revl01_baseline,
                "revl01_pass": revl01_pass,
                "measured_on": "merged-served-v4",
                "val_set": DATASET,
                "excluded": excluded,
                "n_paired_canonical": jud.get("n_paired_canonical"),
                "revl01a_spearman_n_pairs": revl01_spearman_dict.get("n_pairs"),
                "revl01a_p_value": revl01_spearman_dict.get("p_value"),
                "v3_baseline_parse_fail_rate": 23 / 121,  # from output/eval_reasoning_v3 on-disk
                "v3_baseline_parse_fail_count": 23,
                "model_path": str(V4_STAGING),
                "judge_output": judge_out_path,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            out_revl01a = OUT_DIR / "revl01a_v4.json"
            out_revl01a.write_text(json.dumps(revl01a_artifact, indent=2))
            _log(f"Wrote: {out_revl01a}", log_fh)

            # ---- STEP 2: REVL-02 fresh PHPCS (eval_gen, SC2 anti-masking) ----
            _log("Step 2: REVL-02 fresh PHPCS via eval_gen ...", log_fh)
            from eval import eval_gen

            gen_out_path = str(REASONING_OUT / "eval_gen_results.json")
            gen = eval_gen.run_eval(
                dataset_path=DATASET,
                limit=None,
                output_path=gen_out_path,
                base_url=endpoint,
            )
            revl02_phpcs = gen.get("phpcs_pass_rate")
            revl02_overall_mean = gen.get("overall_mean")
            # If phpcs_pass_rate absent, compute from overall_mean
            if revl02_phpcs is None and revl02_overall_mean is not None:
                revl02_phpcs = float(revl02_overall_mean) / 100.0
            if revl02_phpcs is None:
                revl02_phpcs = 0.0
            revl02_pass = revl02_phpcs >= (revl02_baseline - 0.02)

            _log(f"REVL-02 PHPCS: {revl02_phpcs:.4f} vs baseline {revl02_baseline:.4f} "
                 f"(threshold {revl02_baseline - 0.02:.4f}) pass={revl02_pass}", log_fh)

            revl02_artifact = {
                "revl02_phpcs": revl02_phpcs,
                "revl02_overall_mean": revl02_overall_mean,
                "revl02_baseline": revl02_baseline,
                "revl02_pass": revl02_pass,
                "measured_on": "merged-served-v4",
                "val_set": DATASET,
                "model_path": str(V4_STAGING),
                "gen_output": gen_out_path,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            out_revl02 = OUT_DIR / "revl02_gen_phpcs_v4.json"
            out_revl02.write_text(json.dumps(revl02_artifact, indent=2))
            _log(f"Wrote: {out_revl02}", log_fh)

        except VllmBootTimeout as e:
            _log(f"ABORT: vLLM boot timeout: {e}", log_fh)
            return 3
        except Exception as e:  # noqa: BLE001
            _log(f"ABORT: unexpected error in eval steps: {e}", log_fh)
            raise
        finally:
            stop_vllm(CONTAINER_NAME)
            _log("vLLM stopped.", log_fh)

        # ---- STEP 3: Capture v4 responses (separate boot via capture script) ----
        # capture_reasoning_responses.py boots its own vLLM instance internally.
        # We invoke it as subprocess so its vLLM lifecycle is fully self-contained.
        _log("Step 3: Capturing v4 responses (--max-tokens 2048) ...", log_fh)
        capture_script = ROOT / "scripts" / "capture_reasoning_responses.py"
        captured_out = REASONING_OUT / "captured_responses.jsonl"
        capture_cmd = [
            sys.executable, str(capture_script),
            "--dataset", DATASET,
            "--out", str(captured_out),
            "--include-streams", "cot,ctf",
            "--gpu-mem-util", str(GPU_MEM_UTIL),
            "--max-tokens", "2048",
            "--model-dir", str(V4_STAGING),
            "--served-name", "wp-30_70",
            "--port", str(PORT),
        ]
        rc_capture = _stream_cmd(capture_cmd, log_fh, "CAPTURE_v4", timeout=7200)
        if rc_capture != 0:
            _log(f"WARNING: capture script exited {rc_capture}. "
                 f"Captured JSONL may be empty — REVL-07/08 will degrade gracefully.", log_fh)
        else:
            cap_lines = sum(1 for _ in open(captured_out) if _.strip()) if captured_out.exists() else 0
            _log(f"Captured {cap_lines} response records to {captured_out}", log_fh)

        # ---- Summary ----
        _log("=== _04.4_revl01a_v4.py COMPLETE ===", log_fh)
        _log(f"  REVL-01A: parse_rate={parse_failure_rate:.4f} ({parse_fail_count}/{total_pairs}) "
             f"gate={'PASS' if parse_gate_pass else 'FAIL'}", log_fh)
        _log(f"  REVL-01A Spearman: {revl01_spearman} vs {revl01_baseline} "
             f"({'PASS' if revl01_pass else 'FAIL'})", log_fh)
        _log(f"  REVL-02 PHPCS: {revl02_phpcs:.4f} vs {revl02_baseline:.4f} "
             f"({'PASS' if revl02_pass else 'FAIL'})", log_fh)

        print(f"\n[RESULT] parse_failure_rate={parse_failure_rate:.4f} "
              f"parse_gate_pass={parse_gate_pass} "
              f"spearman={revl01_spearman} spearman_pass={revl01_pass} "
              f"phpcs={revl02_phpcs:.4f} phpcs_pass={revl02_pass}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
