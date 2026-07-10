"""Phase 04.4 Plan 08 — REVL-04 wp-bench HARD gate for v4 (nolmhead) candidate.

Gate order per D-IT-09 (fail-fast):
  PRECONDITION: read output/eval_reasoning_v4_nolmhead/revl01a_v4.json
                abort with exit code 7 if parse_gate_pass is not True
                OR parse_failure_rate > 0.05 — BEFORE booting vLLM or wp-bench.
  GATE RUN (only if precondition passes):
    - wp-bench prereqs (npm i -g @wordpress/env; cd wp-bench/runtime && wp-env start)
    - invoke run_eval_reasoning with v4 model + v4 merge_report (served-identity v4)
    - full suite, NOT --wpbench-only (wp-bench never ran on v4; no stale summary.json)
    - results land under output/eval_reasoning_v4_nolmhead/

Harness wiring reused EXACTLY from plan 03 (run_eval_reasoning.py):
  request_timeout 1800s, concurrency 4, enable_thinking=false, max_tokens 2048,
  8-blocker fix chain (npx shim, usercustomize pth, IPv4 NODE_OPTIONS, dummy
  OPENAI_API_KEY, output-file discovery) — all baked in.

Threat mitigations:
  T-0448-01: assert_served_identity called against output/merge_v4_nolmhead/merge_report.json
             (the v4 report) via --merge-report CLI arg (added to run_eval_reasoning.py
             in plan 08 as Rule 2 deviation — backward-compatible, default=MERGE_REPORT_V3).
  T-0448-02: --out-dir output/eval_reasoning_v4_nolmhead (namespaced; no --wpbench-only).
  T-0448-03: HARD gate = reasoning_score >= baseline_score; pass written to results JSON.
  T-0448-04: precondition early-exit (exit 7) before any GPU/wp-bench spend.

Exit codes:
  0  — REVL-04 PASS (reasoning_score >= baseline_score)
  1  — REVL-04 FAIL (reasoning < baseline) or harness error
  3  — vLLM boot timeout
  4  — served-identity check failed
  7  — PRECONDITION FAILED: parse_gate_pass=false in revl01a_v4.json (early-exit,
       wp-bench NOT started — this is the correct designed behavior when upstream
       REVL-01A has not passed the <=5% gate)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── canonical paths ─────────────────────────────────────────────────────────
REVL01A_PATH = PROJECT_ROOT / "output" / "eval_reasoning_v4_nolmhead" / "revl01a_v4.json"
MERGE_REPORT_V4 = PROJECT_ROOT / "output" / "merge_v4_nolmhead" / "merge_report.json"
OUT_DIR = PROJECT_ROOT / "output" / "eval_reasoning_v4_nolmhead"
LOG_DIR = PROJECT_ROOT / "logs" / "phase4.4"

REASONING_MODEL_V4 = (
    "models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4-nolmhead"
)
BASELINE_MODEL = "models/qwen3-30b-wp-30_70-merged-v2"

# Expected baseline score sanity check (plan 03 measured 0.4537; warn only, not gate).
BASELINE_SANITY = 0.4537
BASELINE_SANITY_TOL = 0.02

# ── PRECONDITION ─────────────────────────────────────────────────────────────
EXIT_PARSE_GATE_FAIL = 7   # distinct exit: precondition failed, wp-bench not started


def _check_parse_precondition() -> None:
    """Read revl01a_v4.json; abort with exit 7 if parse gate is not green.

    D-IT-09 fail-fast: the ~2.7h wp-bench must not start on a candidate that
    still fails the <=5% parse gate.  Exit 7 is a designed outcome — NOT an
    error in this script.
    """
    if not REVL01A_PATH.exists():
        print(
            f"PRECONDITION ABORT: {REVL01A_PATH} not found — REVL-01A not yet run.",
            file=sys.stderr,
        )
        sys.exit(EXIT_PARSE_GATE_FAIL)

    data = json.loads(REVL01A_PATH.read_text())
    parse_gate_pass = data.get("parse_gate_pass")
    parse_failure_rate = data.get("parse_failure_rate", 1.0)

    if parse_gate_pass is not True or parse_failure_rate > 0.05:
        msg = (
            f"PRECONDITION ABORT (exit {EXIT_PARSE_GATE_FAIL}): parse_gate_pass={parse_gate_pass}, "
            f"parse_failure_rate={parse_failure_rate:.4f} "
            f"(threshold: <= 0.05).  "
            f"v4 attempt-1 candidate is disqualified upstream at the parse gate "
            f"(D-IT-09 fail-fast).  wp-bench NOT started.  "
            f"FAIL-PATH NOTE: attempt-1 (lm_head excluded, q_proj kept) FAILED; "
            f"next option is attempt-2 (exclude q_proj also, MoE-expert-layers-only "
            f"merge) per D-IT-05; if attempt-2 also fails -> D-IT-02 diagnosis before "
            f"any 04.3 retrain.  No new plan created here."
        )
        print(msg, file=sys.stderr)
        # Write abort record (distinct from 04.4_wp_bench_results.json — that file
        # signals a wp-bench run actually happened and must NOT exist on early-exit).
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        abort_record = OUT_DIR / "04.4_revl04_precondition_abort.json"
        abort_record.write_text(
            json.dumps(
                {
                    "exit_code": EXIT_PARSE_GATE_FAIL,
                    "precondition": "parse_gate_pass",
                    "parse_gate_pass": parse_gate_pass,
                    "parse_failure_rate": parse_failure_rate,
                    "threshold": 0.05,
                    "wp_bench_started": False,
                    "source": str(REVL01A_PATH),
                    "fail_path_note": (
                        "attempt-1 (lm_head excluded, q_proj kept) disqualified; "
                        "attempt-2: exclude q_proj also (MoE-expert-layers-only merge, D-IT-05); "
                        "if attempt-2 fails: D-IT-02 diagnosis before any 04.3 retrain"
                    ),
                },
                indent=2,
            )
        )
        print(f"Abort record written: {abort_record}", file=sys.stderr)
        sys.exit(EXIT_PARSE_GATE_FAIL)

    print(
        f"PRECONDITION PASS: parse_gate_pass={parse_gate_pass}, "
        f"parse_failure_rate={parse_failure_rate:.4f} — proceeding to wp-bench.",
        file=sys.stderr,
    )


# ── WP-BENCH PREREQ SETUP ────────────────────────────────────────────────────

def _ensure_wp_env_prereqs() -> None:
    """Ensure wp-bench prereqs are ready (per REVL-04-COMPLETION.md).

    1. npm install -g @wordpress/env   (global wp-env binary)
    2. cd wp-bench/runtime && wp-env start
       If broken: wp-env destroy && wp-env start
    """
    print("=== wp-bench prereq: npm install -g @wordpress/env ===", file=sys.stderr)
    r = subprocess.run(
        ["npm", "install", "-g", "@wordpress/env"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if r.returncode != 0:
        print(f"WARNING: npm install -g @wordpress/env exit {r.returncode}", file=sys.stderr)
        print(r.stderr[-500:], file=sys.stderr)

    wp_runtime = PROJECT_ROOT / "wp-bench" / "runtime"
    if not wp_runtime.exists():
        print(f"WARNING: {wp_runtime} not found — skipping wp-env start.", file=sys.stderr)
        return

    print("=== wp-bench prereq: wp-env start ===", file=sys.stderr)
    r = subprocess.run(
        ["wp-env", "start"],
        cwd=str(wp_runtime),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        print(
            f"wp-env start failed (exit {r.returncode}) — attempting wp-env destroy && wp-env start",
            file=sys.stderr,
        )
        subprocess.run(["wp-env", "destroy", "--yes"], cwd=str(wp_runtime), timeout=300)
        r2 = subprocess.run(
            ["wp-env", "start"],
            cwd=str(wp_runtime),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r2.returncode != 0:
            print(
                f"WARNING: wp-env start still failed after destroy (exit {r2.returncode}). "
                "wp-bench may fail.",
                file=sys.stderr,
            )
    else:
        print("wp-env start OK.", file=sys.stderr)


# ── HARNESS INVOCATION ───────────────────────────────────────────────────────

def _run_revl04_harness() -> int:
    """Invoke run_eval_reasoning with v4 model paths + v4 merge_report.

    Reuses plan 03's full harness wiring:
      - NOT --wpbench-only (fresh full wp-bench; no stale summary.json to reuse)
      - --merge-report points at the v4 report (T-0448-01: assert_served_identity
        fingerprints the v4 candidate, not v3)
      - All 8-blocker fixes (npx shim, usercustomize pth, IPv4 NODE_OPTIONS,
        dummy OPENAI_API_KEY, output-file discovery) are in run_eval_reasoning.py
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "wpbench_v4.log"

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_eval_reasoning",
        "--reasoning-model",
        REASONING_MODEL_V4,
        "--baseline-model",
        BASELINE_MODEL,
        "--out-dir",
        str(OUT_DIR),
        "--merge-report",
        str(MERGE_REPORT_V4),
    ]

    print(
        f"=== REVL-04 v4: invoking run_eval_reasoning (full fresh wp-bench) ===",
        file=sys.stderr,
    )
    print(f"Command: {' '.join(cmd)}", file=sys.stderr)
    print(f"Log: {log_path}", file=sys.stderr)

    with open(log_path, "w", buffering=1) as log_fh:
        start = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stderr.write(line)
            log_fh.write(line)
        proc.wait()
        elapsed = time.time() - start

    print(
        f"=== run_eval_reasoning exited {proc.returncode} ({elapsed:.0f}s) ===",
        file=sys.stderr,
    )
    return proc.returncode


# ── POST-RUN GATE VALIDATION ─────────────────────────────────────────────────

def _validate_gate_artifact() -> int:
    """Read 04.4_wp_bench_results.json and report HARD gate verdict."""
    wp_results_path = OUT_DIR / "04.4_wp_bench_results.json"
    if not wp_results_path.exists():
        print(
            f"GATE ERROR: {wp_results_path} not found after run — harness error.",
            file=sys.stderr,
        )
        return 1

    data = json.loads(wp_results_path.read_text())
    reasoning_score = data.get("reasoning_score")
    baseline_score = data.get("baseline_score")
    meets_baseline = data.get("meets_baseline")
    passed = data.get("pass")

    # Sanity-check baseline score (warn only; the live comparison is the gate).
    if baseline_score is not None:
        delta = abs(baseline_score - BASELINE_SANITY)
        if delta > BASELINE_SANITY_TOL:
            print(
                f"WARNING: baseline_score={baseline_score:.4f} deviates from expected "
                f"{BASELINE_SANITY} by {delta:.4f} (tolerance {BASELINE_SANITY_TOL}). "
                "Not a gate failure — live comparison governs.",
                file=sys.stderr,
            )
        else:
            print(
                f"Baseline sanity OK: {baseline_score:.4f} ≈ {BASELINE_SANITY} "
                f"(delta {delta:.4f})",
                file=sys.stderr,
            )

    verdict = "PASS" if passed else "FAIL"
    print(
        f"\n=== REVL-04 HARD GATE: {verdict} ===\n"
        f"  reasoning_score = {reasoning_score}\n"
        f"  baseline_score  = {baseline_score}\n"
        f"  meets_baseline  = {meets_baseline}\n"
        f"  pass            = {passed}",
        file=sys.stderr,
    )

    if not passed:
        print(
            "FAIL-PATH NOTE: reasoning_score < baseline_score -> attempt-1 (lm_head excluded) "
            "failed HARD gate; next step is attempt-2 excluding q_proj (MoE-expert-layers-only "
            "merge) per D-IT-05; if attempt-2 also fails -> D-IT-02 diagnosis before any "
            "04.3 retrain.",
            file=sys.stderr,
        )

    return 0 if passed else 1


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main() -> int:
    # STEP 1: Precondition — must be first, before any GPU/wp-bench spend.
    _check_parse_precondition()   # exits with code 7 if gate not green

    # STEP 2: Ensure wp-bench prereqs.
    _ensure_wp_env_prereqs()

    # STEP 3: Run full fresh REVL-04 wp-bench via the proven plan-03 harness.
    harness_rc = _run_revl04_harness()
    if harness_rc not in (0, 1):
        # Non-gate failures (vLLM boot=3, identity=4, etc.) propagate directly.
        return harness_rc

    # STEP 4: Validate gate artifact and surface verdict.
    return _validate_gate_artifact()


if __name__ == "__main__":
    sys.exit(main())
