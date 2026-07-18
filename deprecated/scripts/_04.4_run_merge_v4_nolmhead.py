"""Host launcher for the Tinker v4 staging merge (lm_head excluded) + 3-anchor certification.

D-IT-04 hypothesis: the manual lm_head LoRA stage caused the generation regression in v3
(REVL-04: 0.3716 < 0.4537, 19% parse failures). This launcher reruns the merge with
--exclude-lm-head, writing a DISTINCT v4 candidate so the v3 failure artifacts are preserved
for a clean v4-vs-v3 comparison (Pitfall 5 staging isolation).

D-IT-05 attempt-1 scope: lm_head EXCLUDED, attention q_proj KEPT.
Exactly ONE variable changes vs v3: the manual lm_head LoRA stage is skipped.
MoE per-expert deltas, attention q/k/v/o PEFT merge, per-expert-differ guard, tokenizer
asserts, and the 3 anchor gates all run UNCHANGED.

Pre-flight RAM floor (70 GiB) -> CPU merge (merge_tinker_v3.py --exclude-lm-head, streaming
log) -> consolidated anchors (_04.4_anchors_v3.py --report v4 --staging v4) -> final gate:
lm_head_excluded AND NOT lm_head_applied AND per-expert-differ > 1e-5 (w1/w2/w3) AND
base/tokenizer vocab correct AND all 3 anchors PASS AND v3 report untouched.

Exits non-zero (blocks plan 07) on any failure.
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
LOG = os.path.join(ROOT, "logs", "phase4.4", "merge_v4_nolmhead.log")
REPORT = os.path.join(ROOT, "output", "merge_v4_nolmhead", "merge_report.json")
STAGING = os.path.join(ROOT, "models", "_staging", "qwen3-30b-wp-30_70-reasoning-merged-v4-nolmhead")
# v3 failure artifact paths — must remain UNTOUCHED after this run.
V3_REPORT = os.path.join(ROOT, "output", "merge_v3", "merge_report.json")
V3_STAGING = os.path.join(ROOT, "models", "_staging", "qwen3-30b-wp-30_70-reasoning-merged-v3")
# Canonical (old ckpt-72 merge) — also must remain untouched.
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

    # Record v3 report mtime before the run so we can assert it is untouched after.
    v3_mtime = os.path.getmtime(V3_REPORT) if os.path.exists(V3_REPORT) else None
    canon_mtime = os.path.getmtime(CANONICAL_REPORT) if os.path.exists(CANONICAL_REPORT) else None

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)

    with open(LOG, "a") as log_fh:
        # Step 1: CPU merge with --exclude-lm-head, writing to v4-namespaced paths.
        merge_cmd = [
            sys.executable, MERGE,
            "--exclude-lm-head",
            "--output-dir", STAGING,
            "--report", REPORT,
        ]
        rc = _stream(merge_cmd, log_fh, "MERGE_v4_nolmhead")
        if rc != 0:
            print(f"ABORT: merge exited {rc} (2=RAM,3=canonical-guard,4=broadcast). Plan 07 blocked.",
                  file=sys.stderr)
            return rc

        # Step 2: Run anchors redirected to v4 report + v4 staging (not the v3 paths).
        anchors_cmd = [
            sys.executable, ANCHORS,
            "--report", REPORT,
            "--staging", STAGING,
        ]
        _stream(anchors_cmd, log_fh, "ANCHORS_v4")

    # Final gate (independent of subprocess rc — the anchors update the report in-place).
    if not os.path.exists(REPORT):
        print("ABORT: v4 merge_report.json missing", file=sys.stderr)
        return 5

    rpt = json.load(open(REPORT))
    differ = rpt.get("per_expert_delta_differ_check", {})

    checks = {
        # D-IT-04 core: lm_head must be excluded, not applied.
        "lm_head_excluded": rpt.get("lm_head_excluded") is True,
        "lm_head_not_applied": rpt.get("lm_head_applied") is False,
        # MoE + attention (D-IT-05: q_proj kept).
        "differ_w1": differ.get("w1", 0) > 1e-5,
        "differ_w2": differ.get("w2", 0) > 1e-5,
        "differ_w3": differ.get("w3", 0) > 1e-5,
        "attention_q_proj_changed": rpt.get("attention_q_proj_changed") is True,
        # Stock tokenizer asserts unchanged.
        "base_vocab_151936": rpt.get("base_vocab") == 151936,
        "tokenizer_stock_151669": rpt.get("tokenizer_vocab") == 151669,
        "tokenizer_text_routing": rpt.get("tokenizer_is_stock_text_routing") is True,
        # 3 anchor verdicts (MoE expert weights + expert-block forward; NOT lm_head).
        "tensor_anchor": rpt.get("tensor_anchor") == "pass",
        "fp32_control_anchor": rpt.get("fp32_control_anchor") == "pass",
        "forward_anchor": rpt.get("forward_anchor") == "pass",
        "shards_13": rpt.get("shard_count") == 13,
    }

    # v3 failure artifact must be untouched (T-0446-01).
    v3_untouched = (v3_mtime is None or not os.path.exists(V3_REPORT)
                    or os.path.getmtime(V3_REPORT) == v3_mtime)
    checks["v3_report_untouched"] = v3_untouched

    # Canonical must also be untouched.
    canon_untouched = (canon_mtime is None or not os.path.exists(CANONICAL_REPORT)
                       or os.path.getmtime(CANONICAL_REPORT) == canon_mtime)
    checks["canonical_untouched"] = canon_untouched

    print("\n========== v4 FINAL GATE ==========")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_ok = all(checks.values())
    print(f"  differ={differ}  status={rpt.get('status')}")
    print(f"  merge_type={rpt.get('merge_type')}")
    print(f"  staging={STAGING}")
    print("====================================")

    if not all_ok:
        failed = [k for k, v in checks.items() if not v]
        print(f"RESULT: FAIL — plan 07 blocked. Failed checks: {failed}", file=sys.stderr)
        return 1

    print("RESULT: PASS — v4 staging merge (lm_head excluded, q_proj kept) anchor-certified.")
    print(f"  Ready for plan 07 REVL-01A parse-rate verification.")
    print(f"  v3 failure artifacts preserved at: {V3_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
