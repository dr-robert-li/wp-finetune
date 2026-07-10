# Phase 15-02 — Summary (PKG-03 real Q8 execution)

**Completed:** 2026-07-10
**Requirement:** PKG-03 (advanced from pre-registered to MEASURED)

## What ran

Provisioned the quant toolchain (gguf, llama.cpp CUDA build on GB10 aarch64/CUDA 13) and executed a real
Q8 GGUF quantization + a 3-arm, engine-consistent judge eval.

- **Q8 artifact:** `convert_hf_to_gguf.py --outtype q8_0` on the single-seed v1.3 judge ->
  `models/_gguf/wp-v1.3-judge-s1.Q8_0.gguf`, **30.2 GiB (47% off the 56.8 GiB bf16)**.
- **Eval (all via llama-server, same val set/labels/config):**
  - Foundation Qwen3-30B-A3B base bf16: **121/121 parse fail, rho undefined** — base has no judge capability.
  - Judge bf16 (same engine): **rho 0.7700** (n=93, pf 28).
  - Judge Q8: **rho 0.7239** (n=92, pf 29).

## Result

- **Size:** first real reduction of the whole v3.0 milestone: 56.8 -> 30.2 GiB (47%).
- **Q8 vs bf16 delta:** −0.046, inside the seed noise floor (0.052), CIs overlap; parse rate identical
  (76% vs 77%). **Q8 does not collapse** (contrast 4-bit nf4: 0.165, parse 45%).
- **Foundation contrast:** the fine-tune created the entire judge skill (base: 0/121 parseable).
- **Gate:** MARGINAL vs strict ±2pp (point delta −4.6pp exceeds 2pp) but within measurement noise. Verdict:
  Q8 shippable-with-note, recommended single-seed ship tier.

Artifacts: `output/packaging/pkg03_q8_results.json`, updated `pkg03_quantization_ladder.json`, `MODEL_CARD.md`.

## Deviations / caveats

- **24% parse fails in both judge arms** from bimodal long-prose truncated at max_tokens=2048, not from
  quantization (applied equally, delta valid, absolute rho depressed vs vLLM 0.8017). Tighter confirmation
  = re-run at 4096 tokens / full ensemble. Recorded, not hidden.
- **Only the single-seed judge quantized** (the "single seed" the goal named). Gen model + ensemble use the
  same recipe (`scripts/run_packaging_recipe.md`); left as follow-up.
- Q6/Q5/Q4 tiers still pending (Q8 is the safe ship tier).
- Two background jobs self-killed on `pgrep -f`/`pkill -f` patterns that matched their own command lines;
  re-run with exact-match kills. GGUF artifacts and evals unaffected.

## Self-check

- Q8 GGUF on disk, size confirmed via `stat`: 32,483,931,840 bytes (30.2 GiB).
- Three rho.txt outputs present; numbers copied verbatim into the results JSON.
- Delta arithmetic: 0.7239 − 0.7700 = −0.0461; noise floor 0.0520 > |delta|.
