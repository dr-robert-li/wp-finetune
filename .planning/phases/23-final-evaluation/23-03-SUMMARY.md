---
phase: 23-final-evaluation
plan: 03
subsystem: evaluation
tags: [judge-rho, unmerged-lora, runtime-lora, vllm-lora, gguf-lora, moe-lora, last-lever]

requires:
  - phase: 23-final-evaluation
    provides: "23-02 EXTENSION verdict (shipped-stack Q8 merged, unequivocal_win=false) + exp2_unmerged_lora_rho.json naming-blockage finding"
  - phase: 21-sft-training-generation-judge-models
    provides: "wp-judge-v4-s1 promoted ep3 adapter (output/base21/judge03_s1_adapter, routed-MoE-expert Tinker export)"
provides:
  - "output/eval4/ext_unmerged_preregistration.md -- pre-registered H1 + decision rule (committed BEFORE measurement)"
  - "output/eval4/ext_unmerged_results.json -- machine-readable verdict: H1 REJECTED, both engines' evidence, decision"
  - "scripts/eval4_ext_unmerged_lora_convert.py -- lossless Tinker-export -> vLLM PEFT-convention (base_layer/experts) converter"
  - "scripts/eval4_ext_unmerged_lora_convert_llamacpp.py -- lossless Tinker-export -> llama.cpp GGUF-convention (gate_up_proj/down_proj) converter"
  - "output/eval4/ext_unmerged_llamacpp_compat.patch -- 2 upstream llama.cpp compat fixes (LoraTorchTensor.ndim, ellipsis-expansion bug), local-only, documented for reproducibility"
  - "VERDICT-EVAL4.md section 7 -- unmerged-serving outcome, last-lever-exhausted status"
affects: [24-conditional-gate, 25-conditional-gate, 26-conditional-gate, 27-packaging]

tech-stack:
  added: []
  patterns:
    - "Source-level engine archaeology before spending GPU time: read the installed vLLM/llama.cpp source to derive the EXACT tensor-naming/shape convention a routed-MoE-expert LoRA adapter must satisfy, rather than guess-and-check on the GPU"
    - "In-process scale-0-vs-scale-1 diff gate (llama-server POST /lora-adapters) isolates a LoRA's real effect on the SAME loaded weights, cheaper and more reliable than a two-model-id vLLM diff gate (which proved flaky under prefix-caching/JIT-warmup nondeterminism)"
    - "Multi-prompt diff gates (>=3 distinct prompts) are required for reliability on nondeterministic serving stacks -- a single-prompt diff gate gave a false positive on first boot (Triton JIT warmup) and a false negative on a clean re-boot"

key-files:
  created:
    - output/eval4/ext_unmerged_preregistration.md
    - output/eval4/ext_unmerged_results.json
    - scripts/eval4_ext_unmerged_lora_convert.py
    - scripts/eval4_ext_unmerged_lora_convert_llamacpp.py
    - scripts/eval4_ext_unmerged_lora_rho.py
    - scripts/eval4_ext_unmerged_llamacpp_run.sh
    - output/eval4/ext_unmerged_convert_receipt_s1.json
    - output/eval4/ext_unmerged_convert_receipt_s1_llamacpp.json
    - output/eval4/ext_unmerged_lora_rho_s1.json
    - output/eval4/ext_unmerged_llamacpp_compat.patch
    - output/eval4/ext_unmerged/judge03_s1_adapter_vllm_peft/ (vLLM-convention converted adapter, unmerged, not committed -- disk artifact)
    - output/eval4/ext_unmerged/judge03_s1_adapter_llamacpp_peft/ (llama.cpp-convention converted adapter, disk artifact)
    - output/eval4/ext_unmerged/judge03_s1_lora.gguf (5.1 GiB f16 GGUF LoRA, disk artifact)
    - output/eval4/ext_unmerged/llamacpp_s1/ (capture + rho + diff-gate logs, disk artifact)
    - models/_gguf/wp-v4-base-raw.Q8_0.gguf (37.8 GiB raw unadapted base GGUF, disk artifact)
  modified:
    - output/eval4/VERDICT-EVAL4.md

key-decisions:
  - "H1 (precision-swamping-at-merge-time) REJECTED: llama.cpp unmerged-serving s1 rho = 0.7833, landing on the served-merged ceiling (0.7872, +0.39pp) and 5.25pp below the capture anchor (0.8358) -- despite a dramatic, verified diff-gate confirming the adapter IS correctly applied via ggml_mul_mat_id"
  - "vLLM's nightly FusedMoE3DWithLoRA kernel accepts a correctly-converted adapter (naming/shape verified against source) but does not measurably apply its delta at inference -- recorded blocked_deeper_than_naming, not debugged further (pre-release kernel, out of scope to fix upstream)"
  - "Per pre-registration, s0/s2 capture and ensemble SKIPPED: H1 rejected at the single-seed gate is itself the decision-relevant result -- no further measurement needed. unequivocal_win stays FALSE, v3 pair stays canonical"
  - "Last-lever status: EXHAUSTED. All three pre-registered serving configurations (bf16-vLLM-merged, Q8-llama.cpp-merged, Q8-llama.cpp-UNMERGED) land in the same ~0.78-0.79 band; only the Tinker capture harness (0.8358) sits above it. Whatever separates capture from every served config is not a merge-precision artifact and is out of scope for this milestone"

patterns-established:
  - "Two small, genuine upstream llama.cpp bugs discovered and locally patched (documented as a .patch file, not committed upstream): LoraTorchTensor.ndim missing property, and an ellipsis-expansion off-by-N bug in __getitem__ -- both previously unexercised because no prior LoRA-to-GGUF conversion had exercised a fused-MoE-checkpoint architecture's gate_up_proj chunking path"

requirements-completed: []

coverage:
  - id: D1
    description: "Pre-registration committed before any adapter conversion, vLLM/llama.cpp boot, or capture"
    requirement: "Phase-23-03-EXTENSION"
    verification:
      - kind: other
        ref: "git log: 687aee7 (ext_unmerged_preregistration.md, before d67eee7/f7e1a56 measurement commits)"
        status: pass
    human_judgment: false
  - id: D2
    description: "vLLM path: correctly-derived PEFT-convention adapter conversion (verified lossless via bit-identical w1/w3 lora_A check), loads without naming error, blocked deeper (no measurable effect on 3/3 diff-gate prompts)"
    requirement: "Phase-23-03-EXTENSION"
    verification:
      - kind: other
        ref: "output/eval4/ext_unmerged_lora_rho_s1.json: boot.boot_ok=true, diff_gate.differs_from_base=false (3 prompts)"
        status: pass
    human_judgment: false
  - id: D3
    description: "llama.cpp path: correctly-derived GGUF-convention adapter conversion, in-process diff-gate PASSED (dramatic format change, lora on vs off), full 121-item capture, 0 parse failures, H1 check applied"
    requirement: "Phase-23-03-EXTENSION"
    verification:
      - kind: other
        ref: "output/eval4/ext_unmerged/llamacpp_s1/eval_summary.json: rho_new=0.7833, n=121, parse_fail=0; output/eval4/ext_unmerged_results.json h1_check.h1_confirmed=false"
        status: pass
    human_judgment: false

duration: ~3h (wall clock incl. detached boots/conversions/captures)
completed: 2026-07-15
status: complete
---

# Phase 23 Plan 03: Unmerged Runtime-LoRA Judge Serving Extension Summary

**The last lever tested: serving the v4 judge adapter UNMERGED (native runtime LoRA, verified applying correctly via a dramatic in-process diff gate) still scores only 0.7833 -- essentially identical to every merged-serving figure measured across this milestone (0.7872-0.7877) and 5.25pp below the Tinker capture anchor (0.8358). Precision-swamping-at-merge-time is REJECTED as the explanation for the serving ceiling; the v3 pair stays canonical.**

## Performance

- **Duration:** ~3h wall clock (source archaeology + converter dev ~90min, vLLM boot attempts ~25min, base GGUF conversion ~2.5min, llama.cpp conversion+patches ~20min, serve+diff-gate+121-item capture ~25min)
- **Completed:** 2026-07-15
- **Tasks:** pre-register, vLLM converter+probe, llama.cpp converter+patches+probe, full capture, verdict+docs

## Accomplishments

- Pre-registered H1 (precision-swamping-at-merge-time) and the decision rule BEFORE any measurement.
- **vLLM path:** read the installed nightly build's source (`vllm/lora/layers/fused_moe.py`, `vllm/lora/model_manager.py`, `vllm/lora/lora_model.py`) to derive the exact PEFT convention (`mlp.experts.base_layer` for fused gate-up, `mlp.experts` for down) a routed-MoE-expert LoRA must satisfy for `FusedMoE3DWithLoRA`. Confirmed Tinker's export is losslessly convertible (w1/w3 `lora_A` are bit-identical per layer -- Tinker already shares one A across gate+up, exactly matching what vLLM's kernel needs). The converted adapter **loaded without any naming error** (`MoE model detected. Using fused MoE LoRA implementation.`), resolving the naming blockage from `exp2_unmerged_lora_rho.json`. A robust 3-prompt diff gate then showed 0/3 outputs differ from raw base on a clean boot -- the kernel accepts the adapter but doesn't apply it. Recorded `blocked_deeper_than_naming`; not debugged further (pre-release feature, `0.20.2rc1.dev196`).
- **llama.cpp path:** derived the base checkpoint's own fused naming (`mlp.experts.gate_up_proj`/`down_proj`, verified directly from `model.safetensors.index.json`) and converted Tinker's export to match, materializing the shared-A tensor to full per-expert shape (`ggml_mul_mat_id` gathers by real expert id, unlike vLLM/PyTorch's broadcast semantics -- a broadcast-1 leading dim would have risked out-of-bounds indexing). `convert_lora_to_gguf.py` hit two genuine, previously-unexercised upstream bugs (missing `LoraTorchTensor.ndim`; an ellipsis-expansion off-by-N in `__getitem__`) -- both are one-line, well-understood fixes, patched locally and documented as a `.patch` file for reproducibility (not committed upstream, out of scope).
- Built a fresh raw (unadapted) base Q8_0 GGUF (`models/_gguf/wp-v4-base-raw.Q8_0.gguf`, 37.8 GiB, block-count sanity PASS) and served it with `llama-server --lora`.
- **In-process diff gate** (`POST /lora-adapters` scale 0 vs 1, same loaded weights) gave dramatic, unambiguous confirmation: adapter off produces generic rambling "thinking" text; adapter on produces the exact trained 9-dimension WPCS judge rubric (WPCS Compliance / SQL Safety / Security / Performance / WP API Usage / Code Quality / Dependency Integrity / i18n, each X/10).
- Full 121-item, 8192-token, temp-0 capture: **rho = 0.7833** (n=121, parse_fail=0, CI [0.7134, 0.8346]).
- Applied the pre-registered H1 check: 0.7833 is 5.25pp below the capture anchor (0.8358) and only 0.39pp above the served-merged ceiling (0.7872) -- **H1 REJECTED**. Per the pre-registered stop condition, s0/s2 capture and ensemble were correctly skipped.
- Updated `VERDICT-EVAL4.md` §7 with the full outcome and "last-lever-exhausted" determination.

## The numbers

| Serving configuration | rho (s1) |
|---|---|
| Tinker capture (reference) | **0.8358** |
| bf16-vLLM, merged | 0.7872 |
| Q8-llama.cpp, merged | 0.7877 |
| **Q8-llama.cpp, UNMERGED (this run)** | **0.7833** |

All three served configurations land within a 0.44pp band; only the capture path sits meaningfully above it.

## Task Commits

1. **Pre-registration** — `687aee7` (docs)
2. **Converters + harnesses (vLLM + llama.cpp)** — `d67eee7` (feat)
3. **Results receipts + VERDICT §7** — `f7e1a56` (docs)
4. **SUMMARY + STATE** — (this commit, docs)

## Deviations from Plan

- **[Rule 1 - Bug] vLLM `LoraTorchTensor` missing `.ndim` and off-by-N ellipsis expansion in `convert_lora_to_gguf.py`'s `__getitem__`.** Neither is a wp-finetune codebase bug -- both are genuine, previously-unexercised gaps in the local llama.cpp checkout, hit only because no prior LoRA-to-GGUF conversion had exercised a fused-MoE-checkpoint architecture's `gate_up_proj` chunking path through `LoraTorchTensor`. Root-caused via direct source tracing (not guess-and-check), fixed with two small, correct, well-commented one-line/few-line patches. Left uncommitted upstream (out of scope for this repo); diff captured at `output/eval4/ext_unmerged_llamacpp_compat.patch` for reproducibility.
- **[Rule 1 - Bug] vLLM diff-gate false positive on first boot.** A single-prompt diff gate showed `differs_from_base=true` on the very first boot, then `false` on an identical re-boot/re-probe -- traced to Triton-kernel-JIT-compilation-during-first-inference nondeterminism (visible in `docker logs`: multiple `jit_monitor.py` warnings during exactly that window), not a real adapter effect. Fixed by widening the diff gate to 3 distinct prompts and requiring evidence across all of them before trusting a "no effect" or "has effect" read.

## Issues Encountered

None beyond the deviations above. Teardown verified clean (no `llama-server` process, no stray `ext-unmerged-*`/`exp2-*` docker containers) before writing the verdict receipt.

## Next Phase Readiness

- **Phases 24-27:** judge-only-shipping recommendation is unchanged and now exhaustively tested across every serving configuration this milestone examined: **ship v3's judge pair** (v1.3, Q8 ensemble 0.8056). No further serving-mechanics lever remains to test for v4.
- The capture-vs-served gap (~0.83-0.84 -> ~0.78-0.79) persists identically whether merged or unmerged, across two independent engines. If judge quality is revisited in a future milestone, the productive next step is a gap-closure diagnostic on the serving-vs-capture harness itself (tokenizer/chat-template/sampling parity), not further serving-mechanics experiments -- per `DIAGNOSTIC_SYNTHESIS.md`'s original recommendation, now doubly confirmed.

---
*Phase: 23-final-evaluation*
*Completed: 2026-07-15*

## Self-Check: PASSED

All key files verified present on disk (converters, harnesses, receipts, patch file, VERDICT-EVAL4.md §7). All 3 commit hashes (687aee7, d67eee7, f7e1a56) found in git log. Teardown verified: no llama-server/vllm containers running, GPU idle.
