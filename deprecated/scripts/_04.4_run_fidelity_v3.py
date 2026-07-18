"""Host launcher for the v3 three-layer merge-fidelity gate (plan 04.4-02 Task 2).

Boots the v3 vLLM serve (serve_reasoning_v3_vllm.sh) on GB10, runs the 3 serve preconditions,
then L1 (forward-anchor corroboration, read from plan-01 merge_report — NOT carry-setting),
L2 (24 invalid-PHP sentinel verdict agreement vs Tinker, BLOCKING), L3 (Spearman >= 0.95 on 121
val rows vs Tinker, BLOCKING). Writes output/fidelity_v3/fidelity_report.json with
carry_judge_evidence = (L2 pass AND L3 pass). vLLM is always stopped on exit.

carry_judge_evidence governs ONLY the REVL-01 judge-Spearman carry (plan 04). REVL-02 generation
PHPCS is never carried — plan 04 measures it fresh on merged-served (SC2).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts._p0_vllm_smoke_serve import wait_healthy, stop_vllm, VllmBootTimeout  # noqa: E402
import scripts.fidelity_gate_v3 as fg  # noqa: E402

PORT = 8021
NAME = "wp-reasoning-v3-vllm"
ENDPOINT = f"http://localhost:{PORT}/v1"
SERVE_SH = os.path.join(ROOT, "scripts", "serve_reasoning_v3_vllm.sh")
MERGE_REPORT = os.path.join(ROOT, "output", "merge_v3", "merge_report.json")
OUT_DIR = os.path.join(ROOT, "output", "fidelity_v3")
LOG = os.path.join(ROOT, "logs", "phase4.4", "fidelity_v3.log")
REPORT = os.path.join(OUT_DIR, "fidelity_report.json")


def _log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n"); fh.flush()


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    reuse = os.environ.get("REUSE_SERVE") == "1"   # attach to an already-running warm container
    report = {"status": "running", "endpoint": ENDPOINT, "reuse_serve": reuse}
    rc = 1
    with open(LOG, "a") as fh:
        try:
            if reuse:
                _log("REUSE_SERVE=1 — attaching to warm container (no boot/stop)", fh)
                served = wait_healthy(PORT, NAME)
                _log(f"served healthy: {served}", fh)
            else:
                _log(f"booting v3 serve via {SERVE_SH} on :{PORT}", fh)
                env = dict(os.environ, PORT=str(PORT), CONTAINER_NAME=NAME)
                subprocess.run(["bash", SERVE_SH], cwd=ROOT, env=env, check=True,
                               stdout=fh, stderr=subprocess.STDOUT)
                served = wait_healthy(PORT, NAME)
                _log(f"served healthy: {served}", fh)

            # --- Preconditions (abort on any failure) ---
            pre = fg.run_preconditions(ENDPOINT)
            report["preconditions"] = {
                "served_identity": pre["served_identity"]["ok"],
                "tokenizer_vocab": pre["tokenizer"]["tokenizer_vocab"],
                "tokenizer_len": pre["tokenizer"]["tokenizer_len"],
                "tokenizer_stock_text_routing": pre["tokenizer"]["stock_text_routing"],
                "think_disabled": pre["think"]["ok"],
                "_detail": pre,
            }
            _log(f"preconditions OK: {report['preconditions']['served_identity']}, "
                 f"vocab={report['preconditions']['tokenizer_vocab']}, "
                 f"think={report['preconditions']['think_disabled']}", fh)

            # --- L1 corroboration (forward anchor from plan 01; does NOT set carry) ---
            mr = json.load(open(MERGE_REPORT)) if os.path.exists(MERGE_REPORT) else {}
            report["L1"] = {"layer": "L1_forward_anchor", "source": "merge_v3/merge_report.json",
                            "forward_anchor": mr.get("forward_anchor"),
                            "router_invariant": mr.get("forward_anchor_router_invariant"),
                            "corroboration_only": True}
            _log(f"L1 forward_anchor (corroboration): {report['L1']['forward_anchor']}", fh)

            # --- L2 sentinel (BLOCKING) ---
            _log("L2: generating 24 invalid-PHP sentinel verdicts on merged-served ...", fh)
            report["L2"] = fg.run_l2_sentinel(ENDPOINT, os.path.join(OUT_DIR, "sentinel_merged_served.jsonl"))
            _log(f"L2: agreement {report['L2']['agreement']}/{report['L2']['n']} "
                 f"merged_false_pass={report['L2']['merged_false_pass']} pass={report['L2']['pass']}", fh)

            # --- L3 Spearman (BLOCKING) ---
            _log("L3: generating 121 val judge outputs on merged-served ...", fh)
            report["L3"] = fg.run_l3_spearman(ENDPOINT, os.path.join(OUT_DIR, "spearman_merged_served.jsonl"))
            _log(f"L3: spearman={report['L3']['spearman']} coverage={report['L3']['parse_coverage']} "
                 f"pass={report['L3']['pass']}", fh)

            carry = bool(report["L2"]["pass"] and report["L3"]["pass"])
            report["carry_judge_evidence"] = carry
            if not report["L2"]["pass"]:
                report["l2_failure_diagnosis"] = fg.l2_failure_diagnosis()
            report["status"] = "fidelity_proven" if carry else "fidelity_not_proven"
            rc = 0 if carry else 1
            _log(f"CARRY_JUDGE_EVIDENCE = {carry}  ({report['status']})", fh)
        except VllmBootTimeout as e:
            report["status"] = "serve_boot_failed"; report["error"] = str(e)
            _log(f"BOOT FAIL: {e}", fh); rc = 2
        except fg.PreconditionError as e:
            report["status"] = "precondition_failed"; report["error"] = str(e)
            _log(f"PRECONDITION FAIL: {e}", fh); rc = 3
        except Exception as e:  # noqa: BLE001
            report["status"] = "error"; report["error"] = repr(e)
            _log(f"ERROR: {e!r}", fh); rc = 4
        finally:
            if reuse:
                _log("REUSE_SERVE=1 — leaving container up for next iteration", fh)
            else:
                stop_vllm(NAME)
                _log("vLLM stopped", fh)

    with open(REPORT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport: {REPORT}  | status={report['status']} | "
          f"carry={report.get('carry_judge_evidence')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
