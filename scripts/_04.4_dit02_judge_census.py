#!/usr/bin/env python3
"""D-IT-02 judge-skill-location probe: REVL-01A judge census on single-component variants.

Complements the codegen attribution (MoE carries codegen damage, attention codegen-safe). This
locates JUDGE skill: runs the RC-A-confirmed judge census (parse_failure_rate + overall Spearman,
via the patched enable_thinking=False eval_judge) on attn-only and MoE-only, vs the references
already on disk:
  v3_full   parse 0.0248  Spearman 0.2446   (all components, revl01a_v3_rcA_confirm.json)
  baseline  parse  ~0      Spearman 0.2678   (no reasoning adapter)

Interpretation:
  attn-only ~ v3_full AND moe-only ~ baseline -> judge skill is ATTENTION-borne
     => retrain can cut MoE (codegen-safe) WITHOUT losing judge skill. Clean single-model win.
  moe-only ~ v3_full AND attn-only ~ baseline -> judge skill is MoE-borne
     => cutting MoE for codegen costs judge quality; retrain must balance MoE rank carefully.
  both ~ v3_full (or both ~ baseline) -> distributed / adapter judge-Spearman contribution small.

Caveat: Spearman is one judge axis; the v3 adapter's verdict-policy/format value (Phase 4.3
corrective) is only partly captured. parse_failure_rate is a format-stability proxy. Read both.
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

DATASET = "data/reasoning_dataset/openai_val.jsonl"
OUT_DIR = ROOT / "output" / "eval_reasoning_probe_dit02"
RESULT = OUT_DIR / "dit02_judge_location_result.json"
LOG = ROOT / "logs" / "phase4.4" / "dit02_judge_census.log"
PORT = 8021
GPU = 0.55

V3_FULL_SPEARMAN = 0.2446
V3_FULL_PARSE = 0.0248
BASELINE_SPEARMAN = 0.2678

VARIANTS = [
    ("attn_only", "models/_staging/qwen3-30b-wp-30_70-probe-attn-only", "wp-jc-attn"),
    ("moe_only", "models/_staging/qwen3-30b-wp-30_70-probe-moe-only", "wp-jc-moe"),
]


def _log(msg, fh):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True); fh.write(line + "\n"); fh.flush()


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w") as fh:
        for tag, d, _ in VARIANTS:
            if not (ROOT / d / "config.json").exists():
                _log(f"ABORT: missing {tag}: {d}", fh); return 1

        from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm, VllmBootTimeout
        from eval import eval_judge

        endpoint = f"http://localhost:{PORT}/v1"
        os.environ["EVAL_JUDGE_BASE_URL"] = endpoint
        out = {"references": {"v3_full": {"spearman": V3_FULL_SPEARMAN, "parse": V3_FULL_PARSE},
                              "baseline": {"spearman": BASELINE_SPEARMAN}}, "variants": {}}

        for tag, d, container in VARIANTS:
            _log(f"--- {tag}: boot + judge census ({d}) ---", fh)
            try:
                boot_vllm(str(ROOT / d), container, PORT, GPU)
                served = wait_healthy(PORT, container)
                _log(f"{tag}: served {served}", fh)
                jud = eval_judge.run_eval(
                    dataset_path=DATASET, limit=None,
                    output_path=str(OUT_DIR / tag / "eval_judge_results.json"),
                    base_url=endpoint, output_format="auto", gt_mode="calibrated_canonical",
                )
                excl = jud.get("excluded", {})
                n = jud.get("n_examples", 0) or 0
                pf = excl.get("parse_fail", 0)
                sp = (jud.get("revl01a_overall_spearman_HARD", {}) or {})
                rec = {"parse_fail": pf, "n": n, "parse_rate": (pf / n if n else None),
                       "spearman": sp.get("corr"), "n_pairs": sp.get("n_pairs"), "excluded": excl}
                _log(f"{tag}: parse={pf}/{n} rate={rec['parse_rate']} spearman={rec['spearman']}", fh)
                out["variants"][tag] = rec
            except VllmBootTimeout as e:
                _log(f"{tag}: BOOT TIMEOUT {e}", fh); out["variants"][tag] = {"error": str(e)}
            finally:
                stop_vllm(container); _log(f"{tag}: vLLM stopped", fh)
            (OUT_DIR / tag).mkdir(parents=True, exist_ok=True)

        # Verdict: which variant's Spearman is closer to v3_full vs baseline
        a = out["variants"].get("attn_only", {}).get("spearman")
        m = out["variants"].get("moe_only", {}).get("spearman")
        if a is None or m is None:
            out["verdict"] = "INCOMPLETE — a census failed; see log."
        else:
            # distance to v3_full (judge-rich) vs baseline (judge-floor)
            a_to_v3, a_to_base = abs(a - V3_FULL_SPEARMAN), abs(a - BASELINE_SPEARMAN)
            m_to_v3, m_to_base = abs(m - V3_FULL_SPEARMAN), abs(m - BASELINE_SPEARMAN)
            out["attn_spearman"], out["moe_spearman"] = a, m
            attn_judgey = a >= m  # which single component judges better
            spread = abs(a - m)
            if spread < 0.02:
                out["verdict"] = (f"DISTRIBUTED/SMALL — attn {a:.4f} ~ moe {m:.4f} (spread {spread:.4f}); "
                                  f"adapter judge-Spearman contribution is small / not component-localized. "
                                  f"Cutting MoE unlikely to cost much Spearman-judge skill. Confirm verdict/format "
                                  f"behavior on the retrained candidate.")
            elif attn_judgey:
                out["verdict"] = (f"JUDGE SKILL leans ATTENTION (attn {a:.4f} > moe {m:.4f}) — GOOD: retrain can "
                                  f"cut MoE (codegen-safe) while keeping attention-borne judge skill. Clean single-model path.")
            else:
                out["verdict"] = (f"JUDGE SKILL leans MoE (moe {m:.4f} > attn {a:.4f}) — TENSION: MoE carries BOTH "
                                  f"judge skill AND codegen damage. Retrain must balance MoE rank (can't simply drop it).")
        RESULT.write_text(json.dumps(out, indent=2))
        _log(f"VERDICT: {out['verdict']}", fh)
        _log(f"Wrote: {RESULT}", fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
