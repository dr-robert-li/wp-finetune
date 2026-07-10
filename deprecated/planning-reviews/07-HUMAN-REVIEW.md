# Phase 7 — Router Profiling Human Review & Sign-Off Pack

**Status:** ✅ APPROVED — Task 3 gate closed (see §5 sign-off, 2026-06-19)
**Model:** `models/qwen3-30b-wp-30_70-reasoning-merged-v4` (promoted v1.2 merged checkpoint)
**Reviewer action:** Read this pack, then reply `approved` to close the gate (or describe issues).
**Generated:** 2026-06-15 · **Signed off:** 2026-06-19

---

## 1. What was run

Gradient-free router profiling of the promoted v1.2 merged model on the **matched training-data stimulus** (`data/final_dataset/ratio_30_70/openai_train.jsonl`, the same distribution that produced `base_model_eeff.jsonl`).

| Property | Value |
|---|---|
| Architecture | Qwen3-MoE, 48 layers × 128 experts, top-8 routing |
| Reference pass (D-06 literal) | FULL 30/70 training set — 34,855 examples |
| Stability test | single 10% subsample (~3,485 examples) scored against the full ranking |
| Tokens profiled | 785.8M total (117.4M `wp_gen` + 663.4M `wp_judge`), per-layer ×48 |
| Hardware | NVIDIA GB10, headless `ngc-pytorch` container (`transformers 5.5.0`, CUDA forward-compat) |
| Wall time | 6h 30m (07:40:03 → 14:09:57 UTC), GPU exit rc=0 |
| Baseline integrity | `base_model_eeff.jsonl` **unchanged** (`git diff --stat` clean) — T-07-04 mitigated |

All code was test-certified in Plan 07-01 (76 pytest tests green). This plan ran it and validated the **outputs**.

---

## 2. Gate results (all automated gates PASS)

### ① PROF-03 — Jaccard subsample stability (CI-aware, D-09) — ✅ PASS
- **`jaccard_ci_lower = 0.9426 ≥ 0.94`** — bootstrap lower bound over the 48 per-layer Jaccard values.
- Point estimate: mean **0.9685**, min **0.60**.
- **6/48 layers below the 0.94 point threshold:** L9 / L13 / L14 / L31 / L36 = 0.778, **L35 = 0.60** (worst).
- ⚠️ **Judgment item:** the point gate alone would **FAIL** (6 sub-threshold layers). The CI-aware disposition carries it — this is exactly the gate design the prior journal entry committed to (require the bootstrap lower bound to clear the bar, not the point estimate). The L35 = 0.60 mid-network layer is the single softest spot; it reflects subsample noise on a noisier layer, not a code defect (full-set ranking is the reference and is deterministic).
- If you judge the CI margin (0.9426 vs 0.94 — 0.3pt of headroom) too thin, the D-06 fallback is a re-profile with a larger subsample (e.g. `--subsample 0.25`). My read: PASS is sound; the margin is slim but the disposition is the agreed one.

### ② PROF-04 — Concentration metrics — ✅ computed
| Metric | Total | wp_gen | wp_judge |
|---|---|---|---|
| E_eff mean | 72.58 | **60.69** | 72.65 |
| E_eff max | 98.04 | 88.03 | 98.88 |
| E_eff variance | 132.3 | 96.6 | 136.6 |
- **Generation routes more narrowly than judging** (E_eff 60.7 vs 72.7) — gen concentrates on fewer effective experts, judge spreads wider. Plausible: judging the 7-dimension rubric exercises more of the network than generating a single `wp_*` function.
- CV (across experts) mean **1.2024**, CI [1.1403, 1.2703]. Layer-depth skew (early/late CV) **1.107**.

### ③ D-08 — E_eff delta vs base — ✅ non-empty join (48 matched rows)
- Ratio-key normalization seam validated end-to-end (no silent empty join — T-07-06 mitigated).
- Per-layer delta mean **+2.75**, range **−2.57 … +7.31**. Mostly positive → merged model uses *slightly more* effective experts than base.
- ⚠️ **Judgment item:** the three largest deltas are the **final three layers — L45 +7.07, L46 +7.31, L47 +6.46**. This is a coherent late-layer broadening (not scattered noise). Router was frozen during v1.2 LoRA (D-07/A2), so deltas *should* be modest; a ~7-unit shift on an E_eff base of ~72 is ~10% on the last layers. My read: explainable (LoRA on attention/MLP shifts the hidden states the frozen router sees, concentrated where representations diverge most — the top of the stack), and modest in absolute terms. Flagged for your eye because it's the largest structured effect in the report.

### ④ D-03/D-04 — Protected expert mask — ✅ [48,128] bool + sidecar
- **1,480 experts protected** across 48 layers (conservative co-activation: above per-layer mean in **both** `wp_gen` **and** `wp_judge`).
- Per-layer: **25–40 experts** protected (mean 30.8 / 128).
- Sensitivity (D-04): mean-threshold = **1,480** (chosen, conservative), median = 2,477, top-16-intersection = 595. Spread is sensible for Phase 13 headroom tuning.

### ⑤ PROF-05 / GATE-01 — N/A rationale — ✅ documented
- Single survivor (30/70) from Phase-4 NO_SURVIVORS triage. PROF-05 trivially satisfied (full survivor set profiled). GATE-01 degenerate (no selection among one candidate; no fabricated one-row matrix, Pitfall 5).
- Full prose in `output/profiling/reasoning-merged-v4/routing_report_rationale.md`.

---

## 3. Artifacts for review (all under `output/profiling/reasoning-merged-v4/`)

| File | What to check |
|---|---|
| `concentration_report.json` | D-08 E_eff delta table (48 rows), `jaccard_ci_lower`, CV, skew |
| `jaccard_stability.json` | raw 48-element per-layer Jaccard array |
| `protected_expert_mask.json` | protected set per layer (eyeball plausibility) |
| `protected_expert_mask.npy` | [48,128] bool — the must-not-prune set for Phases 11/13 |
| `sensitivity_table.json` | mean/median/top-K spread |
| `routing_report.jsonl` | per-layer counts + E_eff, ratio `30_70` |
| `routing_report_rationale.md` | PROF-05/GATE-01 rationale |

---

## 4. Two items requiring your judgment (summary)

1. **L35 Jaccard = 0.60** (5 others at 0.778) — carried by the CI gate at a thin 0.3pt margin. Accept the CI disposition, or trigger a D-06 larger-subsample re-profile?
2. **Late-layer E_eff broadening (L45–47 ≈ +7)** — explainable and modest, but the largest structured delta. Accept as a routing-shift finding, or investigate per-expert counts before Phase 11/13 consume the mask?

Everything else is clean and automated-gate-green. **Reply `approved` to sign off, or describe what to investigate.**

---

## 5. Sign-off — council-reviewed disposition (2026-06-19)

**Verdict: APPROVED.** Both judgment items resolved as ACCEPT under the [D-09 CI-aware disposition](07-CONTEXT.md) inherited from [D-V4-10](../04.4-reasoning-eval-adapter-merge-inserted/04.4-D-V4-10-WAIVER.md). SOTA-model council (GPT-5.5 / Claude Opus 4.8 / Gemini 3.1 Pro) unanimous on both items; reasoning that survived all three lanes recorded below.

### ① L35 Jaccard = 0.60 — **ACCEPT, no D-06 re-profile**

Full-set ranking is the reference and is deterministic; PROF-03 subsample-Jaccard is a *methodology-stability check*, not the artifact downstream consumers depend on. The protected mask is built from the full 34,855-example pass, not the subsample. L35 = 0.60 reflects subsample variance on a noisier mid-network layer, not a code defect or a routing pathology.

The CI-aware disposition is the contract committed to *before* seeing results (D-09, codified from D-V4-10 §"Forward hardening"). Re-deciding because 0.9426 vs 0.94 looks thin is the post-hoc bar-move the waiver explicitly named as goalpost-moving. Re-profiling at `--subsample 0.25` is a ~6.5h GB10 run that, by construction (averaging over 48 layers), is overwhelmingly likely to tighten the CI without changing the disposition.

Conservative co-activation (D-03) absorbs single-layer noise: a layer whose ranking jitters under subsampling produces at most marginal membership wobble at the threshold, and the D-04 sensitivity spread (1,480 / 2,477 / 595) hands Phase 13 the levers to interrogate L35's protected set specifically if it ever matters.

### ② Late-layer E_eff broadening L45–47 ≈ +7 — **ACCEPT as routing-shift finding**

Mechanism is well-explained and lawful: frozen router (D-07/A2) + LoRA'd attention/MLP shifts the hidden states the router sees; deeper layers carry the most LoRA-influenced representational divergence (deeper accumulation), so the routing distribution shifts there first. A monotone coherent effect across the final 3 of 48 layers, ~10% on an E_eff base of ~72, is the *expected* functional form — not an artifact. A flag would be scattered ±7 (noise), a single isolated +20 layer (router defect), or *narrowing* with concentrated mass on few experts (routing collapse). None of those were observed.

E_eff is informational per D-02; the artifact Phases 11/13 actually consume is `protected_expert_mask.npy`, which is built from above-per-layer-mean co-activation — invariant to a uniform broadening because both `wp_gen` and `wp_judge` means rise together. Late-layer broadening also means *more* experts engaged, which is the safer direction for a downstream pruner.

Pre-ship sanity check confirmed mask coherence on L45–47 (per-layer protected counts within the 25–40 band reported in §2.④; D-04 sensitivity spread unchanged on these layers). No focal router drift; no defect hypothesis to chase.

### Forward hardening folded into mask sidecar

Before the mask is consumed by Phase 11 (MoE-Sieve) and Phase 13 (AIMER/REAP), the `protected_expert_mask.json` sidecar should carry a `layer_stability_notes` block flagging the six sub-threshold-Jaccard layers (L9 / L13 / L14 / L31 / L35 / L36) so Phase 13 doesn't re-discover the band cold:

```json
"layer_stability_notes": {
  "low_jaccard_layers": [9, 13, 14, 31, 35, 36],
  "interpretation": "subsample-noise band; full-set ranking is the reference and is deterministic. Phase 13 tuning should treat these layers' protected membership as marginally less certain — prefer median-threshold mask (2,477) over conservative mask (1,480) when budgeting prune headroom locally on these layers.",
  "source": "07-HUMAN-REVIEW §5, 2026-06-19 council sign-off"
}
```

This is a non-mutative annotation (does not change `protected_expert_mask.npy` bytes; Phase 13 still consumes the canonical conservative mask as the must-not-prune set). It surfaces the methodology-noise signal alongside the artifact so the next consumer can act on it.

### Forward hardening for Phase 13 planning

The Phase 13 plan should pre-commit (not post-hoc) that L45–47 are *expected* to carry the largest E_eff deltas and that pruning sensitivity on those layers must be evaluated against the median-threshold (2,477) mask in addition to the conservative (1,480). Same D-V4-10 lesson applied one layer up in the decision stack: the CI-aware disposition is the contract, set before measurement.

### Sign-off block

- **Automated verdict (unchanged):** all PROF-03 / PROF-04 / D-08 / D-03/D-04 / PROF-05 / GATE-01 gates green per §2.
- **Disposition:** D-09 CI-aware — `jaccard_ci_lower = 0.9426 ≥ 0.94` carries PROF-03; both judgment items ACCEPTED with mechanistic justification recorded above.
- **Mask:** `protected_expert_mask.npy` ([48,128] bool, 1,480 protected) is immutable from this point and shippable to Phases 11/13. `protected_expert_mask.json` sidecar to be amended with `layer_stability_notes` block above (non-mutative annotation) before Phase 11 consumes.
- **Phase 11 / 13 obligations:** consume canonical conservative mask; apply median-threshold (2,477) as headroom-tuning reference on the low-Jaccard band {9, 13, 14, 31, 35, 36} and on L45–47.

**APPROVED — Dr. Robert Li, 2026-06-19.** Phase 7 closed; Phase 8 (reward infrastructure with D-V4-09 +3.58 judge recalibration inheritance) and Phase 11 (MoE-Sieve, consuming the protected mask) unblocked.
