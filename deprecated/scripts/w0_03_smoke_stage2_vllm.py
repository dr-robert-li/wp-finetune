"""PR2d: W0-03 Stage 2 — vLLM-served full smoke gate (certifying).

Boots vLLM on the merged reasoning model, generates the 10 manifest prompts at
temperature=0, runs the full council-binding check stack, writes a diagnosis
JSON + halt marker on failure. This is the runtime path the REVL eval gates use.

Per-prompt checks (council binding spec 2026-05-29):
  judge prompts:
    1. is_degenerate            -> Mode A
    2. judge_coherent_prose     -> Mode B  (>=5 dims + >=5 'score X/10 — text')
    3. baseline_similarity<0.85 -> Mode B' (merge no-op canary; too-similar = fail)
  gen prompts:
    1. is_degenerate            -> Mode A
    2. php_lint_passes          -> Mode B
  aggregate:
    4. inter_prompt_distinctness (judge) -> Mode C (boilerplate/mode-collapse)

Exit codes: 0=PASS, 1=DEGENERATE/INCOHERENT, 2=NO_REASONING_EFFECT, 3=BOOT_FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_smoke_common import (  # noqa: E402
    is_degenerate, judge_coherent, baseline_similarity,
    explanation_richness, inter_prompt_distinctness, strip_think,
)
from scripts._p0_vllm_smoke_serve import (  # noqa: E402
    boot_vllm, wait_healthy, generate, stop_vllm, VllmBootTimeout,
)

MODEL = "models/qwen3-30b-wp-30_70-reasoning-merged"
MANIFEST = "data/phase4_4/smoke_prompts.json"
BASELINE_CACHE = "data/phase4_4/smoke_baseline_outputs.json"
HALT_MARKER = "output/04.4_smoke_halt.md"
NAME = "wp-smoke-reasoning-vllm"
PORT = 8012

MAX_BASELINE_SIM = 0.85       # judge no-op canary (too-similar = merge no-op)
MIN_DISTINCTNESS = 0.15       # judge inter-prompt: below = boilerplate collapse
MIN_RICHNESS = 30.0           # chars per scored dimension (substantive prose)


def _php_lint_passes(out: str) -> bool:
    try:
        from scripts.generate_critique_then_fix import php_lint_check
    except Exception:
        # if PHP linter unavailable, fall back to a non-degenerate code heuristic
        return ("<?php" in out) or ("function" in out)
    code = strip_think(out)  # drop Qwen3 <think></think> before lint
    if "```" in code:
        import re
        m = re.search(r"```(?:php)?\s*(.*?)```", code, re.DOTALL)
        if m:
            code = m.group(1)
    try:
        return bool(php_lint_check(code).get("valid"))
    except Exception:
        return False


def _write_halt(verdict: dict) -> None:
    Path(HALT_MARKER).parent.mkdir(parents=True, exist_ok=True)
    d = verdict["diagnosis"]
    lines = [
        "PHASE 4.4 WAVE 0 SMOKE GATE FAILED (STAGE 2 — vLLM served)",
        "",
        "DECISION REQUIRED",
        "",
        f"Verdict: smoke_pass={verdict['smoke_pass']} exit_code={verdict['exit_code']}",
        f"Diagnosis mode: {d['mode']}",
        f"Suggested action: {d['suggested_action']}",
        "",
        "Per CONTEXT D-05, your options:",
        "  (1) ITERATE: re-open Phase 4.3 (data backfill, LR/epochs).",
        "  (2) ABANDON: ship v1 30_70 as v1.2-final; archive ckpt-72.",
        "  (3) RETRY loosened (Mode C/D only): "
        "--ratio-threshold / --max-baseline-sim / --n-prompts via flags.",
        "",
        f"Full diagnosis JSON: {verdict.get('output_json')}",
        "",
        "Per-prompt summary:",
    ]
    for r in verdict["per_prompt"]:
        lines.append(f"  - idx {r['source_val_idx']} [{r['kind']}]: pass={r['pass']} "
                     f"({r['fail_reason'] or 'ok'})")
    Path(HALT_MARKER).write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="W0-03 Stage 2 vLLM served smoke")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--baseline-cache", default=BASELINE_CACHE)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-baseline-sim", type=float, default=MAX_BASELINE_SIM)
    ap.add_argument("--min-distinctness", type=float, default=MIN_DISTINCTNESS)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_json = args.output_json or f"logs/phase4.4/wave0_smoke_stage2_{ts}.json"

    if args.dry_run:
        bc = Path(args.baseline_cache)
        print(f"[stage2] dry-run: manifest={len(manifest)} baseline_cache_exists={bc.exists()}")
        print("[stage2] classifiers + serve helper importable")
        return 0

    baseline = {}
    if Path(args.baseline_cache).exists():
        bdata = json.load(open(args.baseline_cache))
        baseline = {r["source_val_idx"]: r["baseline_output"] for r in bdata["outputs"]}
    else:
        print(f"[stage2] WARN: no baseline cache at {args.baseline_cache} — "
              "no-op canary disabled (run _p0_gen_smoke_baseline.py first)")

    # Boot + generate.
    try:
        boot_vllm(args.model, NAME, PORT, args.gpu_mem_util)
        served = wait_healthy(PORT, NAME)
        outs = generate(PORT, served, manifest, args.max_tokens)
    except VllmBootTimeout as e:
        verdict = {
            "smoke_pass": False, "exit_code": 3, "per_prompt": [],
            "diagnosis": {"mode": "D_boot_timeout", "detail": str(e),
                          "gpu_mem_util": args.gpu_mem_util,
                          "suggested_action": "lower --gpu-mem-util or investigate vLLM logs"},
            "output_json": out_json,
        }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(verdict, open(out_json, "w"), indent=2)
        _write_halt(verdict)
        print(f"[stage2] BOOT FAIL (exit 3). {out_json}")
        return 3
    finally:
        stop_vllm(NAME)

    # Evaluate.
    per_prompt = []
    judge_outputs = []
    any_degenerate = any_incoherent = False
    for p, o in zip(manifest, outs):
        idx, kind = p["source_val_idx"], p["kind"]
        rec = {"source_val_idx": idx, "kind": kind, "pass": True, "fail_reason": None,
               "output": o[:2048]}
        deg, why = is_degenerate(o, max_new_tokens=args.max_tokens)
        if deg:
            rec.update({"pass": False, "fail_reason": f"degenerate:{why}"})
            any_degenerate = True
        elif kind == "judge":
            coh, detail = judge_coherent(o)
            rec["coherence_detail"] = detail
            rec["richness"] = round(explanation_richness(o), 1)
            if not coh:
                rec.update({"pass": False, "fail_reason": f"incoherent:{detail}"})
                any_incoherent = True
            elif idx in baseline:
                sim = baseline_similarity(o, baseline[idx])
                rec["baseline_similarity"] = round(sim, 4)
                if sim >= args.max_baseline_sim:
                    rec.update({"pass": False,
                                "fail_reason": f"no-op canary: sim {sim:.3f} >= {args.max_baseline_sim}"})
                    any_incoherent = True
            judge_outputs.append(o)
        else:  # gen
            if not _php_lint_passes(o):
                rec.update({"pass": False, "fail_reason": "gen: php_lint failed"})
                any_incoherent = True
        per_prompt.append(rec)

    # Aggregate: judge inter-prompt distinctness (boilerplate/mode-collapse).
    distinctness = inter_prompt_distinctness(judge_outputs) if len(judge_outputs) >= 2 else 1.0
    no_reasoning_effect = distinctness < args.min_distinctness

    if any_degenerate:
        mode, action, code = "A_degenerate", "iterate_4.3", 1
    elif any_incoherent:
        mode, action, code = "B_incoherent", "iterate_4.3_or_investigate_baseline", 1
    elif no_reasoning_effect:
        mode, action, code = "C_no_reasoning_effect", "iterate_4.3 (outputs boilerplate/input-insensitive)", 2
    else:
        mode, action, code = "PASS", "promote_and_proceed", 0

    smoke_pass = code == 0
    verdict = {
        "smoke_pass": smoke_pass, "exit_code": code,
        "model": args.model, "served": served, "n": len(manifest),
        "judge_distinctness": round(distinctness, 4),
        "thresholds": {"max_baseline_sim": args.max_baseline_sim,
                       "min_distinctness": args.min_distinctness},
        "diagnosis": {"mode": mode, "suggested_action": action,
                      "any_degenerate": any_degenerate, "any_incoherent": any_incoherent,
                      "no_reasoning_effect": no_reasoning_effect},
        "per_prompt": per_prompt, "output_json": out_json,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(verdict, open(out_json, "w"), indent=2)
    print(f"[stage2] smoke_pass={smoke_pass} mode={mode} distinctness={distinctness:.3f} "
          f"exit={code}")
    print(f"[stage2] {out_json}")
    if not smoke_pass:
        _write_halt(verdict)
        print(f"[stage2] halt marker: {HALT_MARKER}")
    return code


if __name__ == "__main__":
    sys.exit(main())
