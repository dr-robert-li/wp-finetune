#!/usr/bin/env python3
"""One monitoring tick for the local-consistency GSPO RL run (Phase 09).

Reads the live rl_metrics.jsonl, inspects the two vLLM judge containers, runs a
cheap judge-quality spot-check against the LOCAL consistency endpoint, and APPENDS
a tagged status block to 09-LOCAL-RL-STATUS-UPDATES.md. It NEVER stops the run.

Invoked once per /loop tick:
    .venv-tinker/bin/python scripts/_rl_status_tick.py

Exit code is always 0 (monitoring must never crash the loop); problems are reported
as text in the appended block and on stdout.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METRICS = Path(os.environ.get("RL_METRICS_PATH", REPO / "output/rl_checkpoints/metrics/rl_metrics.jsonl"))
RUN_LOG = Path(os.environ.get("RL_RUN_LOG", REPO / "output/rl_checkpoints/full_run.log"))
STATUS_DOC = REPO / ".planning/phases/09-gspo-training/09-LOCAL-RL-STATUS-UPDATES.md"
CONSISTENCY_URL = os.environ.get("CONSISTENCY_BASE_URL", "http://localhost:8001/v1")
CONSISTENCY_MODEL = os.environ.get("CONSISTENCY_MODEL", "wp_consistency")

# Fixed spot-check pair: a clearly-good critique should out-score a clearly-wrong one.
_SPOT_PHP = (
    "<?php\nfunction wp_get_user_role($user_id) {\n"
    "    $user = get_userdata($user_id);\n"
    "    return $user->roles[0];\n}\n"
)
_SPOT_GOOD = (
    "The function does not check whether get_userdata() returned false for an invalid "
    "user_id, so accessing ->roles[0] will throw on a missing user. It also assumes the "
    "user has at least one role. Both are real correctness bugs."
)
_SPOT_WRONG = (
    "This function is perfectly safe and uses prepared SQL statements with $wpdb->prepare "
    "to prevent injection, and correctly sanitizes all output with esc_html."
)


def _read_metrics_tail(n: int = 6) -> list[dict]:
    if not METRICS.exists():
        return []
    rows: list[dict] = []
    for line in METRICS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return rows[-n:]


def _container_state(name: str) -> str:
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Status}}"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        return out or "NOT-RUNNING"
    except Exception as e:  # noqa: BLE001
        return f"docker-err: {e}"


def _run_log_signals() -> dict:
    sig = {"warm_start": None, "recent_error": None, "last_step_line": None}
    if not RUN_LOG.exists():
        return sig
    tail = RUN_LOG.read_text(errors="replace").splitlines()[-400:]
    for ln in tail:
        if "WARM START" in ln or "train_mlp=True" in ln:
            sig["warm_start"] = ln.strip()[:200]
        if any(k in ln for k in ("Traceback", "CUDA out of memory", "OOM", "Error", "FAILED")):
            sig["recent_error"] = ln.strip()[:200]
        if "step" in ln.lower() and ("reward" in ln.lower() or "elapsed" in ln.lower()):
            sig["last_step_line"] = ln.strip()[:200]
    return sig


def _judge_quality_spotcheck() -> dict:
    """Score the good/wrong critique pair via the LOCAL endpoint. good should > wrong."""
    res = {"good": None, "wrong": None, "ok": None, "err": None}
    try:
        sys.path.insert(0, str(REPO))
        from scripts.rl_judge_dispatch import score_judge_consistency  # noqa: PLC0415
        res["good"] = score_judge_consistency(
            _SPOT_PHP, _SPOT_GOOD, model=CONSISTENCY_MODEL, base_url=CONSISTENCY_URL)
        res["wrong"] = score_judge_consistency(
            _SPOT_PHP, _SPOT_WRONG, model=CONSISTENCY_MODEL, base_url=CONSISTENCY_URL)
        if res["good"] is not None and res["wrong"] is not None:
            res["ok"] = res["good"] >= res["wrong"]
    except Exception as e:  # noqa: BLE001
        res["err"] = repr(e)
    return res


def main() -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = _read_metrics_tail()
    judge_c = _container_state("wp-v4-judge-vllm")
    cons_c = _container_state("wp-consistency-vllm")
    logsig = _run_log_signals()
    spot = _judge_quality_spotcheck()

    if rows:
        last = rows[-1]
        bd = last.get("reward_breakdown", {}) or {}
        rmin = bd.get("reward_min")
        rmax = bd.get("reward_max")
        nonuniform = (rmin is not None and rmax is not None and rmin != rmax)
        step = last.get("step")
        rmean = last.get("reward_mean")
        kl = last.get("kl_sample_train_v1")
        halt = last.get("halt_reason")
        # trend over the tail
        means = [r.get("reward_mean") for r in rows if r.get("reward_mean") is not None]
        trend = ""
        if len(means) >= 2:
            trend = f"trend {means[0]:.3f}→{means[-1]:.3f} over {len(means)} rows"
        metrics_line = (
            f"step={step} reward_mean={rmean} min={rmin} max={rmax} "
            f"non_uniform={nonuniform} kl_v1={kl} halt={halt} {trend}"
        )
    else:
        metrics_line = "NO METRICS YET (run not started or first step pending)"

    spot_line = (
        f"good={spot['good']} wrong={spot['wrong']} good>=wrong={spot['ok']}"
        + (f" ERR={spot['err']}" if spot["err"] else "")
    )

    block = f"""
### D · {now} — RL status tick
- containers: wp_judge=`{judge_c}` | wp_consistency=`{cons_c}`
- metrics: {metrics_line}
- warm_start: {logsig['warm_start'] or 'n/a'}
- recent_error: {logsig['recent_error'] or 'none'}

### E · {now} — Judge-quality spot-check (local consistency endpoint)
- {spot_line}
- verdict: {'OK — local judge discriminates good vs wrong critique' if spot['ok'] else ('DEGENERATE — good not >= wrong, INVESTIGATE' if spot['ok'] is False else 'endpoint not reachable / no score')}
"""
    with STATUS_DOC.open("a") as f:
        f.write(block)

    print(block)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — never crash the loop
        print(f"status-tick error (non-fatal): {e!r}", file=sys.stderr)
        sys.exit(0)
