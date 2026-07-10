---
phase: 13-lora-merge-pruning
verified: 2026-07-10T05:13:23Z
status: passed_with_notes
score: 6/6 roadmap success criteria verified (1 with a materialization note); 7/7 requirements delivered or explicitly dispositioned
behavior_unverified: 0
overrides_applied: 0
notes:
  - "REQUIREMENTS.md checkbox/table staleness: MERGE-01, PRUNE-01, PRUNE-02, PRUNE-03, PRUNE-04 remain unchecked ([ ]) and 'Pending' in the requirements mapping table, while PRUNE-05/06 were correctly marked [x]/Complete. Confirmed via git log --follow -- .planning/REQUIREMENTS.md: only the 13-06 and 13-07 'docs(...): complete plan' commits touched this file; the 13-01..13-05 commits that delivered MERGE-01/PRUNE-01..04 never updated it. All 5 requirements ARE substantively delivered (see Requirements Coverage table) — this is a ledger-sync gap, not a functional gap."
  - "Roadmap SC2 ('AIMER runs ... at 25%, 50%, 75% ... producing 3 pruning masks') is only literally true for 25% — output/prune/masks/ contains aimer_gen_k96.npy and aimer_judge_k96.npy (k=96, 25%) only; no aimer_*_k64.npy or aimer_*_k32.npy were materialized. The underlying AIMER score arrays that would produce those masks do exist (aimer_scores_{gen,judge}.npy, unconditional per PRUNE-01), and the decision not to materialize the 50%/75% masks is explicitly justified by the bounded-worse-by-monotonicity argument in output/prune/expansion_decision.md (citing Phase 11's own k-sweep measurements at the identical k=64/k=32 keep-counts). No downstream decision needed the unmaterialized mask files."
gaps: []
deferred: []
human_verification: []
---

# Phase 13: LoRA Merge & Pruning (AIMER primary, REAP optional) Verification Report

**Phase Goal:** Merge LoRA adapters into base weights, then run AIMER (primary) and optionally REAP (comparison) at three compression ratios to determine whether WordPress domain specialization creates enough routing concentration for calibration-based pruning to outperform generalized weight-based pruning.
**Verified:** 2026-07-10
**Status:** passed_with_notes
**Re-verification:** No — initial verification

## Goal Achievement

The phase's actual delivered outcome is a **gate-before-remove decision procedure that reached a rigorously-evidenced `no_winner` verdict**, human-approved as `ship_unpruned` on 2026-07-10. This is treated as a first-class, valid terminal state (not an incomplete phase) because: (a) the CONTEXT.md's pre-registered branch rule explicitly anticipates "≤25% or nothing," (b) PRUNE-05's own selection rule is a closed-form procedure whose valid outputs include "no eligible variant," (c) it mirrors the precedent already accepted for Phase 12 (SKIPPED, `optimal_k=full`, marked `[x]` N/A-style in REQUIREMENTS.md), and (d) zero physical weight removal ever running is exactly what PRUNE-03's gate-before-remove contract requires when no variant clears the gate.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All LoRA adapters merged into base weights; merged checkpoint verified equivalent to adapter-on-base | ✓ VERIFIED | `MERGE-01-TRACEABILITY.md`: no unmerged adapters exist (RL rejected, commit `8860e89`; Sieve training-free). Gen checkpoint's `merge_report.json` shows `tensor_anchor: pass, fp32_control_anchor: pass, forward_anchor: pass, anchors_all_pass: true`. All 4 checkpoints (gen + judge s0/s1/s2) live-verified via `AutoConfig` (`num_local_experts=128`, `num_hidden_layers=48`) + `index.json` presence — output reproduced independently below. |
| 2 | AIMER runs at 25/50/75% producing pruning masks (task-agnostic, ~1s, no calibration) | ⚠️ NOTED (see notes) | Score arrays (`aimer_scores_{gen,judge}.npy`, [48,128] float32) exist unconditionally per PRUNE-01. Only the 25% keep-masks were materialized to disk (`output/prune/masks/aimer_{gen,judge}_k96.npy`); 50%/75% masks were not written, with an explicit, evidence-backed bounded-worse-by-monotonicity rationale (`expansion_decision.md`). |
| 3 | REAP optionally runs with WordPress calibration data at same 3 ratios | ✓ VERIFIED (correctly not run) | AIMER@25 failed all 3 gates decisively, making REAP's domain comparison moot per PRUNE-02's own conditional rule. All 6 REAP cells (`reap_{25,50,75}_{gen,judge}.json`) are explicit, machine-readable `skipped: true` dispositions — never a silent gap. |
| 4 | All variants evaluated via gating mask across all 9 dims before any weight removal; comparison table visible before committing | ✓ VERIFIED | `output/prune/comparison_table.md` — full 6-variant x 2-axis table with measured values, bars, pass/fail, and disposition for every cell. `output/prune/gated/` contains all 13 expected records (3 measured AIMER@25 axes + 4 bounded AIMER@50/75 + 6 REAP skip stubs). |
| 5 | Domain specificity: AIMER-vs-REAP expert overlap quantified per layer | ✓ VERIFIED (N/A-disposition) | `output/prune/aimer_reap_overlap_25.json` — explicit documented `skipped` disposition (no REAP keep-mask exists to Jaccard against, since REAP never ran). Correctly not computed, not silently omitted. |
| 6 | Winning method+ratio selected by dimension-level retention, OR explicit no-winner outcome; final model physically pruned with router re-normalization (or documented ship-unpruned close) | ✓ VERIFIED | `selection.json`: verdict `no_winner`, produced by `scripts/prune_selection.py` (not hand-declared), `human_signoff.approved: true`, `decision: ship_unpruned` (Dr. Robert Li, 2026-07-10, verbatim note recorded). `prune_methodology.md` documents the uniform-K physical-surgery mechanics as NOT executed. Confirmed independently: no `models/qwen3-30b-wp-pruned*` directory exists; `apply_physical`/`build_uniform_keep_mask` are called only from the script's own `--self-check` and from `tests/test_prune_physical.py` — never from a production driver path. |

**Score:** 6/6 truths verified (1 carries a non-blocking materialization note), 0 present-but-behavior-unverified.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `MERGE-01-TRACEABILITY.md` | N/A-style closure record for MERGE-01 | ✓ VERIFIED | Present, substantive, cites commit `8860e89`, includes reproducible verification command + output, and a "Final lineage (13-07)" closing section. |
| `scripts/aimer_prune.py` | AIMER weight-norm scorer | ✓ VERIFIED | Exists; produces real score arrays consumed downstream. |
| `output/prune/aimer_scores_{gen,judge}.npy` | [48,128] float32 AIMER scores | ✓ VERIFIED | Present, 24.1K each, consistent shape. |
| `scripts/prune_gated_eval.py` | gate-before-remove driver | ✓ VERIFIED | Real GPU execution confirmed via `logs/prune/13-04_full_gate.log` (sequential vLLM boot/serve/stop per model). |
| `scripts/reap_prune.py` + `tests/test_reap_prune.py` | REAP scorer module, unit-tested | ✓ VERIFIED | `compute_reap_scores` deliberately unexecuted per plan (calibration gated on AIMER passing). |
| `scripts/prune_overlap.py`, `scripts/prune_selection.py`, `scripts/prune_apply_physical.py` | PRUNE-04/05/06 modules | ✓ VERIFIED | All present; `prune_apply_physical.py` self-tested but never invoked in production (see truth 6). |
| `output/prune/gated/*.json` (13 files) | 3 measured + 4 bounded + 6 skipped | ✓ VERIFIED | All 13 present; content spot-checked, dispositions explicit and consistent. |
| `output/prune/comparison_table.md`, `selection.json`, `expansion_decision.md`, `prune_methodology.md` | Selection + closure artifacts | ✓ VERIFIED | All present, substantive, mutually consistent (see Numbers Consistency below). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `aimer_scores_{gen,judge}.npy` (13-01) | `prune_gated_eval.py` (13-04 execution) | AIMER@25 keep-mask built from these scores, consumed for real vLLM gate run | ✓ WIRED | `logs/prune/13-04_full_gate.log` shows real vLLM boot/serve on both checkpoints, judge captures for 3 seeds, all against masked model=wp-30_70. |
| `output/prune/gated/*.json` (13-04, 13-05) | `output/prune/selection.json` (13-06) | `scripts/prune_selection.py` reads all 13 gated records, computes eligibility per variant | ✓ WIRED | `selection.json`'s `per_variant` array reproduces the exact measured/bounded reasons from the gated records verbatim (0.1577/0.1651/0.4463). |
| `selection.json` (13-06 verdict + sign-off) | `13-07` (branch selection) | `no_winner` + `ship_unpruned` sign-off gates whether `prune_apply_physical.py` runs | ✓ WIRED | 13-07-SUMMARY.md explicitly confirms the no-surgery branch was taken; no pruned checkpoint directory was created; `MERGE-01-TRACEABILITY.md` final-lineage section appended. |
| `output/sieve/prune_set_for_phase13.json` protected mask pin | every gated variant's `protected_retained` field | sha256 pin re-verified before every measured arm | ✓ WIRED | Independently re-computed sha256 of both `protected_expert_mask.npy` and `.json` — **exact match** against the pinned `mask_npy_sha256` / `mask_json_sha256` values (verified live in this session, not merely asserted). |

### Hard-Constraint Verification (independent checks run this session)

| Constraint | Check performed | Result |
|------------|------------------|--------|
| Protected mask byte-unchanged, sha-pinned | `sha256sum output/profiling/reasoning-merged-v4/protected_expert_mask.{npy,json}` vs pin in `prune_set_for_phase13.json` | **MATCH** — `659af6eb...` and `ade549e0...` both identical to pin |
| No checkpoint modification | `find models -maxdepth 2 -newer 13-CONTEXT.md -name "*.safetensors"` (empty); `config.json.num_local_experts == 128` on gen + judge-s1 checkpoints | **No modification found** — 0 newer safetensors files; `num_local_experts` still 128 on both checked configs |
| No physical pruning ran anywhere | `grep` for `apply_physical(` / `build_uniform_keep_mask(` call sites outside `--self-check` and `tests/` | **0 production call sites** — only self-check (script's own `__main__`) and `tests/test_prune_physical.py` invoke these functions |
| GB10 sequential serving (one ~60GB model resident at a time) | `logs/prune/13-04_full_gate.log` | **Confirmed** — gen vLLM booted/served/stopped fully before judge-s0 vLLM boots; s0 stopped before s1 boots; s1 stopped before s2 boots. No overlapping model residency. |
| Bars never relaxed to Tinker-native | `bars_used` fields in gated records (`gen_wp_bench_floor: 0.4284`, `judge_ensemble_rho_floor: 0.7555`) | **Consistent with vLLM-measured baselines** per `sanity_gate_recalibration.json` (0.4484−2pp / 0.8075−0.052), never the Tinker-native canonical numbers (0.4616 / n/a) |
| No training of any kind | Reviewed all 7 SUMMARYs + `prune_methodology.md` | No training/gradient step anywhere; AIMER is pure weight-norm read; REAP calibration never executed; masking is inference-time only |

### Behavioral Spot-Checks / Real-Execution Evidence

| Behavior | Command / Evidence | Result | Status |
|----------|---------------------|--------|--------|
| AIMER@25 gate was a real GPU run, not simulated | `logs/prune/13-04_full_gate.log` — vLLM health-check after 499s, 121/121 judge captures ok=121 err=0 per seed x3 | Real inference confirmed, not a stub | ✓ PASS |
| A mid-run JSON-serialization bug did not corrupt the underlying measurement | Traced: `TypeError: Object of type bool is not JSON serializable` occurred in `_write_result` AFTER all GPU captures completed; commit `32ce674` fixed `np.bool_` → JSON-serializable and **rescored from the already-captured raw responses, zero re-serve** (`rescored_from_existing_captures: true` field in `aimer_25_judge.json`) | Numbers are derived from the real captured GPU outputs, not fabricated post-crash | ✓ PASS |
| Test suite for all 6 phase-13 modules | `.venv-tinker/bin/python -m pytest tests/test_aimer_prune.py tests/test_reap_prune.py tests/test_prune_overlap.py tests/test_prune_selection.py tests/test_prune_physical.py tests/test_sieve_ksweep_mask.py -q` | `26 passed in 3.12s` | ✓ PASS |

### Numbers Consistency Check

| Metric | `aimer_25_gen.json` / `aimer_25_judge.json` | `comparison_table.md` | `expansion_decision.md` | `prune_methodology.md` | `selection.json` |
|--------|---|---|---|---|---|
| gen wp_bench | 0.1577 | 0.1577 | 0.1577 | 0.1577 | 0.1577 |
| judge ensemble rho | 0.16510... (displayed 0.1651) | 0.1651 | 0.1651 | 0.1651 | 0.1651 |
| judge parse rate | 0.44628... (displayed 0.4463) | 0.4463 | 0.4463 | 0.4463 | 0.4463 |

**No drift found.** All values trace to the single source (`aimer_25_gen.json` / `aimer_25_judge.json`, 13-04) and are reproduced identically (to displayed precision) everywhere they are cited.

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|---|---|---|---|---|
| MERGE-01 | 13-01 | Merge adapters before pruning | ✓ SATISFIED (N/A-closure) | `MERGE-01-TRACEABILITY.md` — no adapters remained to merge; existing merges independently anchor-verified |
| PRUNE-01 | 13-01, 13-04, 13-05 | AIMER at 25/50/75% | ✓ SATISFIED (25% measured, 50/75% bounded-by-monotonicity, documented) | `aimer_scores_*.npy`, `gated/aimer_{25,50,75}_*.json` |
| PRUNE-02 | 13-02, 13-05 | REAP optional comparison | ✓ SATISFIED (correctly not run; conditional rule applied) | `reap_prune.py`, `gated/reap_*.json` (6 skip stubs) |
| PRUNE-03 | 13-02, 13-04 | Gate-before-remove eval | ✓ SATISFIED | `prune_gated_eval.py`, real GPU logs, `protected_retained: true` on every measured arm |
| PRUNE-04 | 13-03, 13-05 | Domain-specificity overlap | ✓ SATISFIED (N/A-disposition, REAP never ran) | `aimer_reap_overlap_25.json` |
| PRUNE-05 | 13-03, 13-06 | Winner selection rule | ✓ SATISFIED | `prune_selection.py`, `selection.json` (`no_winner`), REQUIREMENTS.md already marked `[x]` |
| PRUNE-06 | 13-03, 13-07 | Physical pruning or documented close | ✓ SATISFIED (ship-unpruned branch) | `prune_apply_physical.py` (unexecuted), `prune_methodology.md`, REQUIREMENTS.md already marked `[x]` |

**Orphaned requirements:** None — REQUIREMENTS.md maps exactly MERGE-01 + PRUNE-01..06 to Phase 13, and every ID appears in at least one plan's `requirements:` frontmatter.

**Ledger gap (non-blocking, flagged):** REQUIREMENTS.md's checklist (`[ ]`) and mapping table ("Pending") were never updated for MERGE-01, PRUNE-01, PRUNE-02, PRUNE-03, PRUNE-04, unlike PRUNE-05/06 which correctly show `[x]`/"Complete". Confirmed via `git log --follow -- .planning/REQUIREMENTS.md`: only the 13-06 and 13-07 "complete plan" commits touched this file. Recommend a follow-up commit updating lines 198-202 to `[x]` with closure annotations mirroring the Phase-12 precedent (e.g., `[x] **MERGE-01** (N/A — no unmerged adapters existed; see MERGE-01-TRACEABILITY.md)`).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | `grep` for TBD/FIXME/XXX across all phase-13 scripts, outputs, and planning docs | — | **0 matches** — no unresolved debt markers |
| `.planning/REQUIREMENTS.md` | 198-202 | Stale `[ ]`/"Pending" entries for delivered requirements | ⚠️ Warning | Ledger inconsistency only; does not affect functional delivery (see Ledger gap above) |

### Human Verification Required

None. The single human-decision checkpoint this phase required (the PRUNE-05/06 sign-off between "prune the winner" vs "ship unpruned") was already executed and recorded verbatim during phase execution: `selection.json.human_signoff` — approver "Dr. Robert Li", date 2026-07-10, mechanism `AskUserQuestion (blocking checkpoint:human-verify, plan 13-06 Task 2)`, decision `ship_unpruned`, with the full approval note preserved.

### Gaps Summary

No blocking gaps. Two non-blocking notes are recorded in frontmatter and above:

1. REQUIREMENTS.md checkbox/table staleness for MERGE-01, PRUNE-01..04 (ledger-sync only — recommend a follow-up commit).
2. AIMER 50%/75% keep-masks were not materialized to `.npy` files (only 25%), a deliberate and well-justified cost-avoidance decision (bounded-worse-by-monotonicity, backed by Phase 11's own k-sweep measurements at identical keep-counts) rather than an oversight — flagged here only because it is a literal (not substantive) partial miss against roadmap SC2's "producing 3 pruning masks" wording.

The phase's actual goal — reaching a rigorously gated, evidence-backed, human-approved pruning decision without ever performing unauthorized weight removal — is fully achieved. The `no_winner` → `ship_unpruned` outcome is a first-class, correctly-executed terminal state of the phase's own pre-registered decision procedure, consistent with the Phase 12 precedent already accepted into REQUIREMENTS.md.

---

*Verified: 2026-07-10T05:13:23Z*
*Verifier: Claude (gsd-verifier)*
