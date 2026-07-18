---
phase: 27
slug: packaging-publication-refresh
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-17
---

# Phase 27 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Seeded from `27-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (repo convention, `tests/`) + per-script `--self-check` flags |
| **Config file** | none dedicated — no `pytest.ini`/root `conftest.py`; run `pytest tests/` directly |
| **Quick run command** | `pytest tests/test_prune_selection.py -x` (per-script: `<script> --self-check`) |
| **Full suite command** | `pytest tests/` |
| **Estimated runtime** | ~60s for `pytest tests/`; served-eval gates are minutes-to-hours (measured, not unit) |

**Convention note (from research):** packaging/conversion scripts in this repo carry a `--self-check`
flag inside the script (see `scripts/prune_apply_physical_v4.py --self-check`) rather than a separate
pytest file, when correctness is a shape/assertion check rather than a unit-testable pure function.
This phase follows that convention — the expert-count sanity check is a self-check embedded in the
conversion driver, not a new pytest file for a single assertion.

---

## Sampling Rate

- **After every task commit:** run the relevant `--self-check` / script directly (conversion sanity;
  gate eval for the tier just produced)
- **After every plan wave:** regenerate the ladder comparison (`pkg4_quantization_ladder.json`) and run
  `pytest tests/` for any touched shared code
- **Before `/gsd-verify-work`:** full round-trip (upload → download → GGUF load → judge smoke) green
- **Max feedback latency:** ~60s for self-checks and `pytest tests/`; served-eval tiers are inherently
  long-running and are gated per-wave, not per-commit

---

## Per-Task Verification Map

Task IDs are assigned by the planner; this map is seeded at the requirement level and refined by
`/gsd-validate-phase` once plans exist.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | 0 | PKG4-01 | T-27-01 | Structural defect (wrong expert count) cannot reach upload | script self-check | `scripts/eval4_ext_gguf_convert.sh <merged_dir> <out.gguf>` (extended) | ✅ block-count / ❌ W0 expert-count | ⬜ pending |
| TBD | TBD | 1 | PKG4-01 | — | N/A | integration (served) | `scripts/_pkg_gguf_eval_run.sh <gguf> <alias> <out_dir> <port>` | ✅ reuse as-is | ⬜ pending |
| TBD | TBD | 0 | PKG4-01 | T-27-01 | Shared-expert quant type independently verified, not inherited | GGUF metadata inspection | new self-check reading `gguf.GGUFReader` tensor `type` for `shared_expert.*` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | PKG4-02 | — | N/A | integration (measured) | `scripts/_pkg_gguf_eval_run.sh` per tier + `scripts/relabel/eval_relabel.py` | ✅ harness / ❌ W0 bands file | ⬜ pending |
| TBD | TBD | 2 | PUB4-01 | T-27-02 | Publish is an explicitly-gated final step, after local validation | integration (live HF) | `scripts/pub4_validate_upload.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Extend `scripts/eval4_ext_gguf_convert.sh` (or a v4-specific copy) with an **expert-count sanity
      check** — the block-count check exists; the expert-count check does not. A 224-expert checkpoint
      is exactly the case where its absence would bite.
- [ ] New self-check for **shared-expert quant-type independent verification** (per-tensor GGUF type
      inspection post-quantize). `llama-quantize`'s `tensor_allows_quantization()` applies no
      shared-expert special case (verified read, `src/llama-quant.cpp:288-355`), so this check asserts
      the *expected uniform* behavior rather than assuming a different precision.
- [ ] `output/pkg-v4/gate1_baseline_v4.json` — a Gate-1 baseline **on the shipped Q8/llama.cpp stack**.
      None exists for the pruned v4 checkpoint; the closest number (s1 rho 0.8134) is bf16-**vLLM** and
      explicitly flagged non-comparable in `selection_v4.json`.
- [ ] `scripts/pub4_validate_upload.py` — extract the Phase-18 PUB-03 round-trip logic (API listing +
      GGUF load + judge smoke) into a standalone re-runnable script. Only the output receipt
      (`output/packaging/pub03_validation_receipt.json`) survives from Phase 18, not a reusable driver.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| HF repo visibility + card renders correctly on the Hub | PUB4-01 | Hub-side rendering and visibility are a web UI state, not a scriptable assertion | Open `https://huggingface.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf`, confirm public, card renders, YAML frontmatter parsed (tags/license shown) |
| Card passes the operator-only bar (LOCKED DECISION 2) | PUB4-01 | Editorial judgment — "is this operator-first, not a pipeline narrative" is not machine-checkable | Read the card top-to-bottom: no phase history, no MoE-Sieve/AIMER/k-sweep methodology, no RL/Tinker history. Links out to GitHub for methodology. Compare tone to `README.md`, NOT to the v3 card. |
| Upload push authorization | PUB4-01 | Project discipline: publish is a human-authorized final step (v3 PKG-04 precedent) | Human confirms local conversion + ladder gates + validation are complete and reviewed BEFORE the publish task runs |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s for self-checks (served-eval gates exempt — inherently long-running)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
