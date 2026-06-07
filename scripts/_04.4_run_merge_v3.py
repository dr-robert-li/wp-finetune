"""Host launcher for the Tinker v3 staging merge + 3-anchor certification (Plan 04.4-01 Task 3).

Pre-flight RAM floor (70 GiB) -> CPU merge (merge_tinker_v3.py, streaming log) -> consolidated
anchors (_04.4_anchors_v3.py) -> final gate: per-expert-differ > 1e-5 (w1/w2/w3) AND base/
tokenizer vocab == 151936 AND all 3 anchors PASS. Exits non-zero (block plan 02) on any failure.

CPU-only; sequential model loads keep peak RAM ~one model (~57 GiB). Canonical dir is NEVER
touched (merge writes to _staging/, refuses canonical without --force-canonical).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGE = os.path.join(ROOT, "scripts", "merge_tinker_v3.py")
ANCHORS = os.path.join(ROOT, "scripts", "_04.4_anchors_v3.py")
LOG = os.path.join(ROOT, "logs", "phase4.4", "merge_v3.log")
REPORT = os.path.join(ROOT, "output", "merge_v3", "merge_report.json")
STAGING = os.path.join(ROOT, "models", "_staging", "qwen3-30b-wp-30_70-reasoning-merged-v3")
CANONICAL_REPORT = os.path.join(ROOT, "models", "qwen3-30b-wp-30_70-reasoning-merged", "merge_report.json")
RAM_FLOOR_GIB = 70.0
MERGE_TIMEOUT = 7200


def _free_ram_gib() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)
    return float("inf")


def _stream(cmd, log_fh, label) -> int:
    print(f"\n>>> {label}: {' '.join(cmd)}", flush=True)
    log_fh.write(f"\n===== {label} @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} =====\n")
    log_fh.flush()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
            log_fh.flush()
        p.wait(timeout=MERGE_TIMEOUT)
    except subprocess.TimeoutExpired:
        p.kill()
        print(f"TIMEOUT: {label} exceeded {MERGE_TIMEOUT}s", file=sys.stderr)
        return 124
    return p.returncode


def main() -> int:
    free = _free_ram_gib()
    if free < RAM_FLOOR_GIB:
        print(f"ABORT: free RAM {free:.1f} GiB < {RAM_FLOOR_GIB} GiB floor. "
              f"Close Chromium / heavy apps and retry.", file=sys.stderr)
        return 2
    print(f"[preflight] free RAM {free:.1f} GiB >= {RAM_FLOOR_GIB} GiB floor OK")

    canon_mtime = os.path.getmtime(CANONICAL_REPORT) if os.path.exists(CANONICAL_REPORT) else None
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as log_fh:
        rc = _stream([sys.executable, MERGE], log_fh, "MERGE")
        if rc != 0:
            print(f"ABORT: merge exited {rc} (2=RAM,3=canonical-guard,4=broadcast). Plan 02 blocked.", file=sys.stderr)
            return rc
        rc = _stream([sys.executable, ANCHORS], log_fh, "ANCHORS")

    # Final gate (independent of subprocess rc).
    if not os.path.exists(REPORT):
        print("ABORT: merge_report.json missing", file=sys.stderr)
        return 5
    rpt = json.load(open(REPORT))
    differ = rpt.get("per_expert_delta_differ_check", {})
    checks = {
        "differ_w1": differ.get("w1", 0) > 1e-5,
        "differ_w2": differ.get("w2", 0) > 1e-5,
        "differ_w3": differ.get("w3", 0) > 1e-5,
        "base_vocab_151936": rpt.get("base_vocab") == 151936,
        "tokenizer_vocab_151936": rpt.get("tokenizer_vocab") == 151936,
        "lm_head_applied": rpt.get("lm_head_applied") is True,
        "tensor_anchor": rpt.get("tensor_anchor") == "pass",
        "fp32_control_anchor": rpt.get("fp32_control_anchor") == "pass",
        "forward_anchor": rpt.get("forward_anchor") == "pass",
        "shards_13": rpt.get("shard_count") == 13,
    }
    canon_untouched = (canon_mtime is None or not os.path.exists(CANONICAL_REPORT)
                       or os.path.getmtime(CANONICAL_REPORT) == canon_mtime)
    checks["canonical_untouched"] = canon_untouched

    print("\n========== FINAL GATE ==========")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_ok = all(checks.values())
    print(f"  differ={differ}  status={rpt.get('status')}")
    print(f"  staging={STAGING}")
    print("================================")
    if not all_ok:
        print("RESULT: FAIL — D-05 hard block. Diagnose before any retry; plan 02 stays blocked.", file=sys.stderr)
        return 1
    print("RESULT: PASS — v3 staging merge anchor-certified. Awaiting human approval (Task 3 gate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
