"""PR2c: W0-03 Stage 1 — CPU in-process degenerate-only pre-flight.

Cheap fail-fast BEFORE paying the ~900s vLLM boot cost. Loads the merged
reasoning model on CPU, generates n=3 short completions, runs is_degenerate
ONLY. No coherence, no divergence, no baseline comparison, no PASS verdict.

Verdict: DEGENERATE_DETECTED -> write halt marker + exit 1; else exit 0 (proceed).

Council scope lock (2026-05-29): degenerate-only, n=3, 128 tokens, 180s guard.
A clean Stage 1 does NOT mean smoke pass — only "no gross collapse, proceed to Stage 2".
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

from scripts._p0_smoke_common import is_degenerate  # noqa: E402

DEFAULT_MODEL = "models/qwen3-30b-wp-30_70-reasoning-merged"
DEFAULT_MANIFEST = "data/phase4_4/smoke_prompts.json"
HALT_MARKER = "output/04.4_smoke_halt.md"

STAGE1_PROMPTS = 3
STAGE1_MAX_TOKENS = 128
STAGE1_TIMEOUT_SEC = 180


def _write_halt(mode: str, detail: str, outputs: list[dict]) -> None:
    Path(HALT_MARKER).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "PHASE 4.4 WAVE 0 SMOKE GATE FAILED (STAGE 1 — CPU degenerate pre-flight)",
        "",
        "DECISION REQUIRED",
        "",
        f"Stage: 1 (CPU degenerate-only pre-flight)",
        f"Diagnosis mode: {mode}",
        f"Detail: {detail}",
        "",
        "Per CONTEXT D-05, your options:",
        "  (1) ITERATE: reasoning adapter is collapsing — re-open Phase 4.3 "
        "(backfill data, adjust LR/epochs).",
        "  (2) ABANDON: ship v1 30_70 as v1.2-final; archive ckpt-72 unmerged.",
        "",
        "Stage 1 caught gross collapse on the merged model BEFORE vLLM boot cost.",
        "This is the 4.3 4-bit-collapse class of failure on a bf16 substrate — investigate.",
        "",
        "Per-prompt outputs (truncated):",
    ]
    for o in outputs:
        lines.append(f"  - idx {o['source_val_idx']} [{o['kind']}]: "
                     f"degenerate={o['degenerate']} ({o['reason']})")
        lines.append(f"    {o['output'][:300]!r}")
    Path(HALT_MARKER).write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="W0-03 Stage 1 CPU degenerate pre-flight")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--n-prompts", type=int, default=STAGE1_PROMPTS)
    ap.add_argument("--max-tokens", type=int, default=STAGE1_MAX_TOKENS)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate manifest + imports only; no model load")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    prompts = manifest[: args.n_prompts]
    print(f"[stage1] model={args.model} n={len(prompts)} max_tokens={args.max_tokens} "
          f"timeout={STAGE1_TIMEOUT_SEC}s")

    if args.dry_run:
        print(f"[stage1] dry-run: manifest OK ({len(manifest)} prompts), is_degenerate importable")
        return 0

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)

    print(f"[stage1] loading {args.model} on CPU bf16 ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": "cpu"}, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()

    t0 = time.time()
    outputs = []
    any_degenerate = False
    for p in prompts:
        if time.time() - t0 > STAGE1_TIMEOUT_SEC:
            print(f"[stage1] WALL-CLOCK GUARD: exceeded {STAGE1_TIMEOUT_SEC}s; treating as collapse")
            _write_halt("A_degenerate", f"stage1 timeout >{STAGE1_TIMEOUT_SEC}s", outputs)
            return 1
        inputs = tok(p["instruction"], return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=args.max_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
            )
        out = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        bad, reason = is_degenerate(out, max_new_tokens=args.max_tokens)
        outputs.append({"source_val_idx": p["source_val_idx"], "kind": p["kind"],
                        "degenerate": bad, "reason": reason, "output": out})
        print(f"  idx {p['source_val_idx']} [{p['kind']}]: degenerate={bad} ({reason})")
        if bad:
            any_degenerate = True

    if any_degenerate:
        _write_halt("A_degenerate", "Stage 1 detected collapse on >=1 prompt", outputs)
        print("[stage1] DEGENERATE_DETECTED -> halt (exit 1). See", HALT_MARKER)
        return 1
    print(f"[stage1] no gross collapse in {time.time()-t0:.0f}s -> proceed to Stage 2 (exit 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
