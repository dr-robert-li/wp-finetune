#!/usr/bin/env python3
"""D-IT-02 attribution probe: wp-bench subset on single-component merges.

Runs a FIXED wp-bench subset (WPBENCH_LIMIT) on four models to attribute the RC-B codegen
regression to attention vs MoE:

  baseline_v2  models/qwen3-30b-wp-30_70-merged-v2                    (anchor: good codegen, ~0.4537 full)
  v3_full      models/_staging/...-reasoning-merged-v3               (anchor: all components, ~0.3716 full)
  probe_attn   models/_staging/qwen3-30b-wp-30_70-probe-attn-only    (attention deltas only)
  probe_moe    models/_staging/qwen3-30b-wp-30_70-probe-moe-only     (MoE per-expert deltas only)

The two anchors confirm the subset reproduces the known full-set gap before trusting the split.
Reuses run_eval_reasoning._wpbench_with_boot (all 8 wp-bench blocker fixes + thinking-off shim,
applied symmetrically). All variants serve as wp-30_70.

ATTRIBUTION (on subset overall score):
  gap = baseline - v3_full                          (total damage reproduced on this subset)
  attn_damage = baseline - probe_attn
  moe_damage  = baseline - probe_moe
  A component "carries" the damage if its damage >= 50% of gap. Reports attention / MoE / both /
  neither, plus the additivity check (attn_damage + moe_damage vs gap).
This guides the Phase-4.3 retrain (which modules to drop/shrink). It does NOT re-enable MoE-subset
salvage (already ruled out by the uniform delta-norm result).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WPBENCH_LIMIT = int(os.environ.get("WPBENCH_LIMIT", "30"))
os.environ["WPBENCH_LIMIT"] = str(WPBENCH_LIMIT)  # ensure set for the imported harness

OUT_DIR = ROOT / "output" / "eval_reasoning_probe_dit02"
RESULT = OUT_DIR / "dit02_attribution_result.json"
LOG = ROOT / "logs" / "phase4.4" / "dit02_probe_wpbench.log"

MODELS = [
    ("baseline_v2", "models/qwen3-30b-wp-30_70-merged-v2", "wp-probe-baseline"),
    ("v3_full", "models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v3", "wp-probe-v3full"),
    ("probe_attn", "models/_staging/qwen3-30b-wp-30_70-probe-attn-only", "wp-probe-attn"),
    ("probe_moe", "models/_staging/qwen3-30b-wp-30_70-probe-moe-only", "wp-probe-moe"),
]


def _log(msg, fh):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True); fh.write(line + "\n"); fh.flush()


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w") as fh:
        # Preflight: all model dirs exist
        for tag, d, _ in MODELS:
            if not (ROOT / d / "config.json").exists():
                _log(f"ABORT: model dir missing for {tag}: {d}", fh)
                return 1
        _log(f"WPBENCH_LIMIT={WPBENCH_LIMIT} (per phase: knowledge + execution)", fh)

        from scripts.run_eval_reasoning import _wpbench_with_boot

        scores = {}
        for tag, d, container in MODELS:
            # _run_wpbench writes its tmp config + log to OUT_DIR/<tag>/ but does not mkdir it;
            # a missing parent raises FileNotFoundError that the harness mislabels as
            # "wp-bench CLI not on PATH". Create the per-tag dir up front.
            (OUT_DIR / tag).mkdir(parents=True, exist_ok=True)
            _log(f"--- {tag}: boot + wp-bench subset ({d}) ---", fh)
            res = _wpbench_with_boot(str(ROOT / d), container, tag, 0.55, OUT_DIR)
            ran = res.get("ran")
            overall = res.get("wpbench_score")
            _log(f"{tag}: ran={ran} overall={overall} scores={res.get('scores')} "
                 f"err={res.get('error')}", fh)
            scores[tag] = res

        def ov(tag):
            v = scores.get(tag, {}).get("wpbench_score")
            return float(v) if v is not None else None

        b, v3, a, m = ov("baseline_v2"), ov("v3_full"), ov("probe_attn"), ov("probe_moe")
        result = {
            "wpbench_limit": WPBENCH_LIMIT,
            "overall": {"baseline_v2": b, "v3_full": v3, "probe_attn": a, "probe_moe": m},
            "scores_full": {k: scores[k].get("scores") for k in scores},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if None in (b, v3, a, m):
            result["verdict"] = "INCOMPLETE — one or more wp-bench runs failed; see log."
        else:
            gap = b - v3
            attn_dmg = b - a
            moe_dmg = b - m
            result["gap_baseline_minus_v3full"] = gap
            result["attn_damage"] = attn_dmg
            result["moe_damage"] = moe_dmg
            result["additivity_check"] = (attn_dmg + moe_dmg) - gap
            half = 0.5 * gap if gap > 0 else 0.0
            attn_carries = gap > 0 and attn_dmg >= half
            moe_carries = gap > 0 and moe_dmg >= half
            if gap <= 0.02:
                result["verdict"] = (f"SUBSET DID NOT REPRODUCE GAP (gap={gap:.4f} <= 0.02) — "
                                     f"subset too small/easy; rerun with larger WPBENCH_LIMIT before trusting split.")
            elif attn_carries and moe_carries:
                result["verdict"] = ("BOTH components contribute — damage is distributed; retrain "
                                     "should reduce rank across attention AND MoE (or accept).")
            elif attn_carries:
                result["verdict"] = ("ATTENTION carries the codegen damage — retrain dropping/shrinking "
                                     "attention LoRA (q/k/v/o); MoE deltas largely codegen-safe.")
            elif moe_carries:
                result["verdict"] = ("MoE carries the codegen damage — retrain with lower MoE rank / "
                                     "fewer expert modules; attention deltas largely codegen-safe.")
            else:
                result["verdict"] = ("NEITHER single component reaches half the gap — likely an "
                                     "interaction effect; treat as entangled, lean retrain-conservative or accept.")
        RESULT.write_text(json.dumps(result, indent=2))
        _log(f"VERDICT: {result['verdict']}", fh)
        _log(f"overall: baseline={b} v3_full={v3} attn={a} moe={m}", fh)
        _log(f"Wrote: {RESULT}", fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
