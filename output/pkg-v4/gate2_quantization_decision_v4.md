# Gate 2 — Quantization Decision (v4, judge-only) (PKG4-02)

**Date:** 2026-07-17
**Decision:** Quantization is WARRANTED — but the ROADMAP's stated reason is void. This document
re-derives the warrant honestly before the Q6/Q5 ladder descends.

## The inherited rationale is VOID

The ROADMAP/REQUIREMENTS justification for Gate 2 reads: "Q8 GGUF **pair** conversion... pair bf16
**134 GiB** > GB10 121 GB." That rationale does **not apply to v4 and is VOID.** The gen role was
retired as a deliverable on 2026-07-15 (`PROJECT.md:14`) — there is no gen model and therefore no
**pair**. Phase 27 ships exactly **one** checkpoint: the pruned v4 judge at
`models/Qwen3.6-35B-A3B-judge-v4-pruned-k224`, measured at **60 GB** bf16 on disk
(`conversion_receipt_v4.json` `source_bf16_gb_on_disk: 59.49`, `selection_v4.json`
`size.pruned_bf16_gb`). The host has **121 GiB** unified memory with ~114 GiB free at rest
(`27-RESEARCH.md` § Environment Availability). **60 fits 121 with room to spare** — the old "pair
serving" / "bf16 pair conversion" constraint that forced quantization under the ROADMAP's original
framing simply does not exist for a single judge-only checkpoint. Concrete proof this is not
theoretical: the **f16 GGUF itself (~57.10 GiB) served cleanly on this host with no OOM**
(`gate1_f16_baseline_v4.json` `f16_anchor_result: SUCCEEDED`) — a much larger artifact than the bf16
checkpoint would need, and it still fit.

The old warrant is dead. Gate 2 rests on new grounds below.

## The real warrant (three grounds, each with evidence)

**1. Distribution size.** An operator pulls this artifact over a network, not across a local NVMe.
Smaller is strictly better for that operator regardless of whether it fits the host that built it.
The measured f16 GGUF master is **57.1023 GiB** and the measured Q8_0 is **30.3746 GiB**
(`conversion_receipt_v4.json` `artifacts.f16.size_gib` / `artifacts.q8_0.size_gib`) — both real
byte counts from the produced files, not projections. Roughly halving distribution size is worth
pursuing on its own.

**2. Operator memory budget.** The point of shipping a GGUF at all is to run on hosts that are NOT
this project's 121 GiB GB10. Q8 roughly halves the resident footprint versus f16/bf16 — the
difference between "loads on a dev box with a single consumer GPU" and "needs a GB10-class machine."
The 121 GiB figure is OUR build/eval constraint; it says nothing about the machines the operators this
model ships to will actually have. Smaller resident footprint is squarely for them, independent of
whether the source checkpoint itself would have fit unquantized.

**3. Measured-lossless precedent, and an honest fresh measurement that does NOT repeat it.** v3's Q8
measured Q8==bf16 within noise (delta −0.0044, `output/packaging/pkg03_quantization_ladder.json` Q8
rung) — the precedent this project has relied on for "Q8 is safe." Gate 1 here is anchored to f16
(`anchor: f16_gguf_llamacpp`, no OOM fallback needed), so a v4-native Q8-vs-f16 comparison WAS
actually made, not assumed. It does **not** repeat v3's zero-cost result: `ladder_q8.json` shows Q8
costs **1.507pp** of Spearman rho against this checkpoint's own f16 master (delta −0.01507), passing
the inclusive −2pp bar with only **0.493pp** of slack. Q8 is quantifiably **not** lossless on this
surgically-pruned 224-expert MoE — a real degradation, not zero, and it is recorded as such rather
than rounded down to match the v3 story. The warrant for continuing to Q6/Q5 does not lean on a
false "costs nothing" claim; it leans on grounds 1-2 (a smaller, more portable artifact is worth
pursuing) plus the fact that even a non-zero-cost Q8 still cleared the pre-registered band on the
one measurement that exists. Whether Q6/Q5 clear it too is exactly what the ladder below measures —
this document does not presume the answer.

## Exclusions carried forward

**No uniform 4-bit nf4** (PKG4-02 explicit prohibition). Carrying forward the v3 collapse tombstone:
`models/qwen3-30b-wp-30_70-merged-v2-4bit` — bitsandbytes nf4 double-quant quantized the router
weights at the same uniform width as the routed experts; a sparse MoE router cannot tolerate that
blur and the checkpoint produced degenerate output regardless of adapter
(`output/packaging/pkg03_quantization_ladder.json` `Q4-nf4` entry). No nf4 tier is produced or
measured in this phase.

**AWQ W4A16** is out of scope. It is a vLLM/activation-aware quantization artifact path, not a GGUF
ship — this phase ships one GGUF file to one HF repo, and AWQ is not that file format.

## Honest execution status

**Measured, this phase:** Gate 1 f16 baseline (`gate1_f16_baseline_v4.json`); Q8 rung
(`ladder_q8.json`), including the concurrent-sequence CUDA-backend smoke. Both ran on the real
toolchain (`~/llama.cpp/build/bin`, this host, this checkpoint) — nothing here is pre-registered and
pending toolchain provisioning, unlike v3's Gate 2 doc (`output/packaging/gate2_quantization_decision.md`),
which had no local quantization toolchain at write time and had to defer Q8/Q6/Q5 to a future run.

**Not yet measured, at the time this document is written:** Q6_K and Q5_K_M. Both are pre-registered
in `pkg4_quantization_ladder.json` before either is quantized (`floor_frozen_utc` predates any Q6/Q5
byte). Q8 already consumed 1.507pp of the 2pp budget, leaving only 0.493pp — Q6/Q5 are **expected**
to be tight or to fail the band, but that expectation is not treated as a result; both rungs are
measured, not assumed, in the tasks that follow this document.

**Inherited, not re-derived here:** the nf4 collapse tombstone above (Phase 4.3 result, unpruned
architecture — still directly applicable since the router-quantization mechanism it documents is
architecture-general, not specific to expert count).
