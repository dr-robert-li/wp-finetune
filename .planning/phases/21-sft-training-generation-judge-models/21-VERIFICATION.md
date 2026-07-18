---
phase: 21-sft-training-generation-judge-models
verified: 2026-07-14T12:45:00Z
status: passed
score: 6/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
gaps: []
deferred: []
carry_forward_phase_23:
  - item: "EVAL4-01 A/B verdict consumes both Phase 21 recorded misses"
    detail: "GEN-03 wp-bench 0.372 (CI-lower 0.2847 < 0.4286 floor) and JUDGE-03 rho 0.7872 vLLM-served / 0.8160 capture-ensemble (both CI-lower < 0.85/0.87 targets) are Phase 21's committed measurements. Phase 23's Success Criterion 3 applies the SAME pre-registered criteria mechanically against these numbers -- Phase 21 does not re-litigate pass/fail, it hands Phase 23 the measured baseline."
  - item: "Epoch-sweep option (ep1/ep2 gen checkpoints) preserved as an unexplored lever"
    detail: "wp-gen-v4-manifest.json preserves all 3 per-epoch sampler checkpoints (ep1/ep2/ep3), not just the promoted ep3. 21-05-SUMMARY.md explicitly flags 'less SFT may mean less damage' as a candidate lever if the milestone needs to clear the wp-bench floor -- an epoch-sweep re-eval is possible WITHOUT retraining, using the already-preserved ep1/ep2 sampler checkpoints."
  - item: "Judge relabel-campaign re-open condition: unmet, not evaluated"
    detail: "judge03_rho.json records the V4-RERUN-ROADMAP discretion-item-2 re-open condition verbatim: condition (a) saturated-below-target is MET; condition (b) a gap-closure diagnostic (capacity/loss-shape/data-cleaning, mirroring output/relabel/gap_closure_summary.json) has NOT been run on Qwen3.6-35B-A3B. Both are required jointly -- since (b) is unmet, no relabel-campaign re-open is triggered. This diagnostic is flagged as the natural next investigation if the milestone still needs the judge targets after Phase 23's verdict."
---

# Phase 21: SFT Training — Generation & Judge Models Verification Report

**Phase Goal:** Both the generation and judge pathways are fine-tuned on the new base using the existing
reasoning-mix and relabel data (no regeneration), and each clears its pre-registered acceptance bar — or the
miss is recorded as a valid, measured outcome
**Verified:** 2026-07-14T12:45:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Critical Framing Applied

Per the phase goal's explicit text ("or the miss is recorded as a valid, measured outcome"), GEN-03 and
JUDGE-03 are graded on whether the acceptance-bar comparison was **honestly measured, CI-aware, and
recorded** — not on whether the bar was cleared. Both did miss their pre-registered bars. Both misses are
verified below as **VERIFIED (recorded valid miss)**, matching the phase's own success contract. This is not
grade inflation: the underlying numbers (0.372 vs 0.4286; 0.7872/0.8160 vs 0.85/0.87) are reproduced from the
receipts unchanged, and both are flagged plainly in the score narrative and forwarded to Phase 23.

## Goal Achievement

### Observable Truths

| # | Truth (by requirement ID) | Status | Evidence |
|---|-------|--------|----------|
| 1 | **GEN-01** — Data-format/renderer/LR decision recorded; rendered-example spot-check shows no spurious empty `<think></think>`; max tokenized length < 64K | ✓ VERIFIED | `output/base21/gen01_format_decision.json`: `renderer_name="qwen3_5_disable_thinking"` (registry-sourced), `resolved_lr=4.99e-4` with documented supersession rationale tracing to ROADMAP.md's Phase-4.3 note, `output_router_logits_disposition` investigated and recorded as "N/A at Tinker's abstraction layer" (grep evidence cited), `max_tokenized_len=7851 < 64000` (`len_under_64k=true`), `empty_think_injected=false`, `n_train_examples=560`, `n_val_examples=136` |
| 2 | **GEN-02** — Generation model SFT completes (MoE-only LoRA r32, frozen router, Tinker auto-LR) on the reused reasoning-mix with decreasing loss + per-epoch sampler checkpoints; terse format-stability gate measured | ✓ VERIFIED | `output/base21/gen02_run.json` "full" block: `full_ok=true`, `loss_first=7.973 → loss_last=1.458` (monotone decrease), `rank=32`, `train_mlp=true/train_attn=false/train_unembed=false`, 210 steps / 3 epochs. `output/tinker/wp-gen-v4-manifest.json`: 3 checkpoints (ep1/ep2/ep3), all with `sampler_path`, `promoted="wp-gen-v4-ep3"`. `output/base21/gen02_fs_gate.json`: terse rate 0.0 @temp0 / 0.83% @temp0.7, both `pass=true` against the 10%/15%-Wilson-upper gate |
| 3 | **GEN-03** — Merged gen model serves over vLLM; wp-bench measured CI-aware vs 0.4286 floor (pass or recorded miss) | ✓ VERIFIED (recorded valid miss) | `output/base21/gen03_merge.json`: `merge_ok=true`, 240/240 module-count guard, `base_vs_merged_differs=true` — verified NON-vacuous by reading `gen03_merge_log.txt` directly: merged output is a real PHP function body, base output is a real distinct completion (not an empty-string vacuous pass — the CR-02 bug found in code review does not corrupt this historical receipt). `output/base21/gen03_wpbench.json`: `wpbench_overall=0.372`, `wpbench_ci_lower=0.2847 < floor 0.4286` → `pass=false`. Fresh raw-base anchor measured at 0.4897 (ABOVE floor, `material_shift=false`) — the fresh-floor escape hatch does not apply; `floor_source="inherited"`, floor not swapped. Disposition recorded plainly, not forced/retried |
| 4 | **JUDGE-01** — Raw pre-SFT base's judge-format-compliance parse-fail rate measured on ≥20 real generations and recorded vs the 18% community anchor, before any judge-training result is read | ✓ VERIFIED | `output/base21/judge01_format_smoke.json`: `n_prompts=30`, `n_parse_fail=30` (`parse_fail_rate=1.0`), `community_anchor_rate=0.18`, `vs_anchor="above"`, `max_tokens=2048` (generous, truncation-safe per plan). Timestamp (Jul 14 01:54) precedes every judge-training-RESULT read: `judge_capture_s{0,1,2}.jsonl` (10:19–11:00) and `judge03_rho.json` (12:06) — satisfies the plan's documented "before bulk judge training [result is read]" intent (JUDGE-02's training itself ran concurrently in Wave 2, which the plan explicitly designs for; JUDGE-01 gates the Wave-4 *read*, not the Wave-2 *train*) |
| 5 | **JUDGE-02** — 3-seed relabel-SFT (seeds {1,0,2}) completes on the new base reusing v1.3 human-relabeled targets verbatim; all sampler checkpoints preserved | ✓ VERIFIED | `output/base21/judge02_run.json`: `all_seeds_complete=true`, `relabel_reuse=true`, `label_source="data/reasoning_dataset/openai_train_relabel_v1.jsonl"`. All 3 `output/tinker/wp-judge-v4-s{1,0,2}-manifest.json` confirmed present, each `base_model="Qwen/Qwen3.6-35B-A3B"`, 3 per-epoch checkpoints each (9 total, none pruned), each with a `promoted` name + resolvable `sampler_path` |
| 6 | **JUDGE-03** — Judge rho measured on both the cheap Tinker-capture path (per-seed + 3-seed ensemble) and the literal vLLM-served 8192-cap path; compared CI-aware against pre-registered targets (>0.85 single / >0.87 ensemble); miss recorded as valid outcome if unmet | ✓ VERIFIED (recorded valid miss) | `output/base21/judge03_capture_rho.json`: 3 seeds at `max_tokens=8192`, `parse_fail=0` for all 3 (n=121 each), `best_single_seed.rho=0.8358` (seed 1), `ensemble_median.rho=0.8160`. `output/base21/judge03_rho.json`: promoted seed (s1) merged (240/240 guard, `base_vs_merged_differs=true` — verified NON-vacuous via `judge03_merge_serve_run.log`: real distinct merged/base text), served at 8192-cap MAX_MODEL_LEN=16384, `vllm_served_single_seed.rho=0.7872`, `ci_lower=0.7125 < 0.85` → `single_seed_pass=false`; `ensemble_figure.ci_lower=0.7563 < 0.87` → `ensemble_pass=false`; `overall_pass=false`; `disposition="valid_recorded_miss"`; re-open condition recorded verbatim with `condition_met=false` (diagnostic (b) not yet run on this base) |

**Score:** 6/6 truths verified (0 present-but-behavior-unverified; GEN-03 and JUDGE-03 verified as *recorded valid misses*, which the phase goal explicitly defines as a passing outcome when honestly measured)

### No-Regeneration / Data-Reuse Check

| Check | Result |
|---|---|
| `data/reasoning_dataset/openai_train.jsonl` (gen mix) mtime | May 25 15:44 — predates Phase 21 execution window (Jul 13–14) entirely; not touched |
| `data/reasoning_dataset/openai_train_relabel_v1.jsonl` (judge relabel targets) mtime | Jul 3 12:32 — predates Phase 21 (v1.3 artifact, per plan's explicit reuse discretion item 2); not touched |
| `data/relabel_v1/labels.json` mtime | Jul 3 12:23 — same, not touched |
| `git diff --exit-code scripts/tinker_reasoning_data.py scripts/tinker_reasoning_sft.py` | Clean — v3 originals genuinely untouched; v4 exists only as non-destructive siblings |

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `scripts/tinker_reasoning_data_v4.py` | v4 data adapter (BASE_MODEL + resolved renderer) | ✓ VERIFIED | Exists; `BASE_MODEL="Qwen/Qwen3.6-35B-A3B"` |
| `scripts/tinker_reasoning_sft_v4.py` | v4 SFT driver | ✓ VERIFIED | Imports `BASE_MODEL, RENDERER_NAME, TRAIN_PATH, VAL_PATH, build_datasets` from `tinker_reasoning_data_v4` |
| `scripts/build_base21_moe_probe_adapter.py` | MoE merge probe builder | ✓ VERIFIED | `train_mlp=True` grep confirmed; drives `merge_adapter.py` as subprocess |
| `output/base21/gen01_format_decision.json` | GEN-01 receipt | ✓ VERIFIED | All asserted fields present and pass their own acceptance predicates |
| `output/base21/moe_merge_probe.json` | MoE merge-path proof | ✓ VERIFIED | `merge_ok=true`, 240/240 guard, gap_resolution documents the routed-expert merge-path fix (2026-07-13), cross-verified by `moe_merge_ground_truth.json` (Tinker-sampler-vs-vLLM byte comparison, `verdict_pass=true`) |
| `output/tinker/wp-gen-v4-manifest.json` | Gen Tinker manifest | ✓ VERIFIED | 3 checkpoints, promoted `wp-gen-v4-ep3`, all sampler paths present |
| `output/base21/gen02_run.json` | Gen SFT receipt | ✓ VERIFIED | smoke + full blocks both present and internally consistent |
| `output/tinker/wp-judge-v4-s{1,0,2}-manifest.json` (x3) | Per-seed judge Tinker manifests | ✓ VERIFIED | All 3 present, new base, promoted, sampler-resolvable |
| `output/base21/judge02_run.json` | 3-seed run receipt | ✓ VERIFIED | `all_seeds_complete=true`, provenance recorded |
| `scripts/smoke_judge_format_base21.py` | Raw-base judge-format smoke | ✓ VERIFIED | Calls `parse_judge_scores`; parses cleanly |
| `output/base21/judge01_format_smoke.json` | JUDGE-01 baseline receipt | ✓ VERIFIED | 30 prompts, rate=1.0, anchor comparison recorded |
| `output/base21/gen03_merge.json` | GEN-03 merge receipt | ✓ VERIFIED | Guard + real base-vs-merged text diff confirmed via log |
| `output/base21/gen03_wpbench.json` | GEN-03 wp-bench receipt | ✓ VERIFIED | Full 344-test suite, CI-aware, fresh-anchor conditional logic exercised and recorded |
| `output/base21/judge03_capture_rho.json` | Cheap-path per-seed + ensemble rho | ✓ VERIFIED | 3 seeds + ensemble, all at 8192 cap, `parse_fail=0` each |
| `output/base21/judge03_rho.json` | Final JUDGE-03 receipt | ✓ VERIFIED | vLLM-served rho + CI-aware verdict + disposition, both methodologies labeled distinctly |
| `models/Qwen3.6-35B-A3B-gen-v4-merged` | Merged gen model (~67 GiB) | ✓ VERIFIED | Referenced as `served_model_dir` in `gen03_wpbench.json`; wp-bench ran against it |
| `models/Qwen3.6-35B-A3B-judge-v4-s1-merged` | Merged promoted judge seed | ✓ VERIFIED | Referenced as `served_model_dir` in `judge03_rho.json` |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `tinker_reasoning_sft_v4.py` | `tinker_reasoning_data_v4.py` | `from tinker_reasoning_data_v4 import BASE_MODEL, RENDERER_NAME, TRAIN_PATH, VAL_PATH, build_datasets` | ✓ WIRED | Single source of v4 base + renderer, confirmed by grep |
| `build_base21_moe_probe_adapter.py` | `merge_adapter.py` | subprocess call with `--config-path config/train_config_v4.yaml` | ✓ WIRED | Confirmed by grep + `moe_merge_probe.json` receipt showing a real merge executed |
| `tinker_reasoning_sft_v4.py --stage full` | `wp-gen-v4-manifest.json` | incremental per-epoch manifest write | ✓ WIRED | Manifest contains all 3 epochs' sampler paths, consumed downstream by GEN-03's merge |
| `serve_base20_vllm.sh` (raw base) | `parse_judge_scores` | `smoke_judge_format_base21.py` generation loop | ✓ WIRED | 30 real generations produced, all routed through the parser (100% fail, not a wiring failure — a measured compliance result) |
| `wp-judge-v4-s{1,0,2}` sampler | `capture_judge_responses_tinker.py --max-tokens 8192` | v4-aware `--base-model`/`--renderer` flags | ✓ WIRED | Flags present and reachable (grepped); captures produced non-empty JSONLs, 0 parse fails per seed |
| promoted judge seed sampler | `serve_base20_vllm.sh` (8192 cap) → `eval_relabel.py` | merge → serve → capture → score | ✓ WIRED | `judge03_rho.json.vllm_served_single_seed` is a real measured rho, not a stub |
| `wp-gen-v4-manifest.json` promoted sampler | `run_eval_reasoning._run_wpbench` | `merge_adapter.py` → merged dir → `serve_base20_vllm.sh` | ✓ WIRED | Full pipeline exercised; 344-test wp-bench score produced |

### Data-Flow Trace (Level 4)

Not applicable in the conventional sense (no UI/component rendering dynamic state) — this phase's "data flow"
is the training→merge→serve→eval pipeline, verified end-to-end above via direct receipt inspection and raw
log excerpts (not just presence of JSON keys). Two specific vacuous-pass risks were checked and ruled out:

| Risk | Where it could hide | Check performed | Result |
|---|---|---|---|
| `base_vs_merged_differs=true` vacuously true from empty merged output (code-review CR-02) | `gen03_merge.json`, `judge03_rho.json`, `moe_merge_probe.json` | Read raw subprocess logs (`gen03_merge_log.txt`, `judge03_merge_serve_run.log`) for the actual generated text on both sides | Both merged and base outputs are real, non-empty, substantively different text in every case — none of the three historical receipts were corrupted by the CR-02 bug (which was found and fixed AFTER these runs, for future re-runs only) |
| Judge-format parse-fail rate inflated by truncation | `judge01_format_smoke.json` | `max_tokens=2048` recorded (plan requires ≥2048) | Not a truncation artifact — genuine 100% raw-base non-compliance |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `scripts/smoke_judge_format_base21.py` | 89 | `"placeholder"` in a docstring | ℹ️ Info | Describes the WR-04 fix (infra-error tracking); not a stub or unresolved debt marker. No `TBD`/`FIXME`/`XXX` found anywhere across the 11 phase-modified scripts checked |

No debt-marker gate violations. No unreferenced `TBD`/`FIXME`/`XXX` in any Phase 21 file.

### Code Review Findings (21-REVIEW.md / 21-REVIEW-FIX.md)

A post-execution code review found 3 CRITICAL + 6 WARNING issues in the reusable merge/serve helper
scripts (all in the "second serve" diff-verification pattern, plus CI-reproducibility and idempotency
gaps). All 9 were fixed in `21-REVIEW-FIX.md` (commits `c036487` through `3608b42`), verified by AST syntax
checks and `pytest tests/test_merge_adapter_moe_routing.py` (2 passed, 2 skipped — re-confirmed independently
below). Per both documents' explicit statement, **no historical Phase 21 receipt was rewritten** — the fixes
protect only future re-runs (notably Phase 23's EVAL4-01 and Phase 27's packaging, which reuse these same
scripts). This verification independently confirmed the two receipts most exposed to CR-02 (the empty-output
vacuous-diff bug) were NOT actually corrupted by it, by reading the raw generation text in the run logs rather
than trusting the boolean flag alone (see Data-Flow Trace above).

### Independent Test Run

```
$ python3 -m pytest tests/test_merge_adapter_moe_routing.py -x -q
..ss
2 passed, 2 skipped in 0.57s
```

Matches the expected 2 passed / 2 skipped (the 2 skips require `tinker_cookbook`, unavailable in this conda
env — documented as pre-existing/expected in `21-REVIEW-FIX.md`).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| GEN-01 | 21-01-PLAN.md | Data-format/renderer/LR decision + spot-check | ✓ SATISFIED | `gen01_format_decision.json` |
| GEN-02 | 21-02-PLAN.md | Gen SFT (MoE-only r32, frozen router, auto-LR) | ✓ SATISFIED | `gen02_run.json`, `wp-gen-v4-manifest.json` |
| GEN-03 | 21-05-PLAN.md | wp-bench CI-aware vs floor (pass or recorded miss) | ✓ SATISFIED (recorded miss) | `gen03_merge.json`, `gen03_wpbench.json` |
| JUDGE-01 | 21-04-PLAN.md | Raw-base format-compliance smoke before bulk read | ✓ SATISFIED | `judge01_format_smoke.json` |
| JUDGE-02 | 21-03-PLAN.md | 3-seed relabel-SFT reusing v1.3 labels | ✓ SATISFIED | `judge02_run.json`, 3 manifests |
| JUDGE-03 | 21-06-PLAN.md | Judge rho CI-aware vs targets (pass or recorded miss) | ✓ SATISFIED (recorded miss) | `judge03_capture_rho.json`, `judge03_rho.json` |

All 6 requirement IDs declared in Phase 21 plans are marked `[x]` Complete in `.planning/REQUIREMENTS.md`
(lines 399-404, 428-436) — no orphaned or under-claimed requirements found. No requirement mapped to Phase 21
in REQUIREMENTS.md is missing a plan claim.

### Human Verification Required

None. All Phase 21 truths are grounded in machine-readable receipts, raw subprocess logs, and an independently
re-run test suite — no visual/UX/real-time judgment call remains open.

### Gaps Summary

No gaps. All 6 must-haves (GEN-01/02/03, JUDGE-01/02/03) are verified against ground-truth receipts, not
SUMMARY.md narrative. The phase's two pre-registered acceptance-bar misses (GEN-03 wp-bench, JUDGE-03 rho)
are exactly the outcome the phase goal explicitly designates as valid when honestly measured and recorded —
both are CI-aware, both compare against a freshly-measured control (raw new-base anchor for GEN-03; the
discretion-item-2 re-open condition for JUDGE-03), and neither was papered over or silently retried. A
post-execution code review found and fixed 9 real defects in the reusable merge/serve/bootstrap
infrastructure (3 of them measurement-integrity-affecting); this verification independently confirmed none of
those defects corrupted the historical Phase 21 receipts by reading the raw generation logs rather than
trusting the boolean summary fields.

### Carry-Forwards for Phase 23 (EVAL4-01)

See `carry_forward_phase_23` in the frontmatter. Summary:
1. Phase 23's A/B verdict is the mechanical application of the SAME pre-registered criteria against the SAME
   numbers Phase 21 already measured (GEN-03: 0.372/0.2847; JUDGE-03: 0.7872 served / 0.8160 capture-ensemble)
   — Phase 21 does not pre-judge Phase 23's verdict, only supplies the measured inputs.
2. All 3 gen per-epoch sampler checkpoints (ep1/ep2/ep3) are preserved on Tinker and in the manifest — an
   epoch-sweep re-eval (less SFT, potentially less codegen damage) is available without retraining if the
   milestone needs to chase the wp-bench floor further.
3. The judge relabel-campaign re-open condition is explicitly unmet (gap-closure diagnostic not yet run on
   this base) — recorded, not triggered. If Phase 23 or a later phase decides the judge targets must be
   chased further, that diagnostic (mirroring `output/relabel/gap_closure_summary.json`) is the documented
   next step, not a fresh relabel campaign taken speculatively.

---

_Verified: 2026-07-14T12:45:00Z_
_Verifier: Claude (gsd-verifier)_
