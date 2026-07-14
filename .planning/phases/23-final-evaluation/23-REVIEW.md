---
phase: 23-final-evaluation
plan: 01
reviewed: 2026-07-15T08:12:00+10:00
files_reviewed:
  - scripts/build_eval4_comparison.py
status: issues
findings_count: 1
severity_breakdown:
  critical: 0
  high: 0
  medium: 0
  low: 1
---

# Phase 23 Plan 01: Code Review — build_eval4_comparison.py

## Scope

Single new script (`scripts/build_eval4_comparison.py`, 359 lines), pure-Python synthesis over
existing JSON receipts under `output/base21/`. Reviewed for: numbers sourced from receipts (not
hand-typed literals), CI-aware criteria logic, mechanical criteria application, fail-closed
behavior on comparability gaps.

## Method

- Read the full script end to end.
- Cross-checked every number written into `comparability_audit.json` and
  `eval4_final_comparison.json` against its claimed source receipt, byte-for-byte (9 figures
  checked: `0.4897`/raw-base overall, `0.4381`/`0.3295`/`0.5504` ep1, `0.372`/`0.2847`/`0.4753`
  ep3, `0.4022`/`0.2924`/`0.5122` v4b, `0.7872053287497484`/`0.7125475563884955` judge served-s1,
  `0.8160452097775477`/`0.756291877837854` judge capture-ensemble, `0.8358149892119933` judge
  capture-s1-reference — all match their `output/base21/...` receipts exactly).
- Independently re-ran `_bootstrap_ci_lower` on the raw-base results file outside the script;
  reproduced `ci_lower=0.3812, ci_upper=0.5983` exactly (deterministic, `bootstrap_seed=1337`).
- Traced `emit_verdict()`'s hard-fail path (missing audit / un-reconciled comparability gap →
  `sys.exit(2)`/`sys.exit(3)`) and confirmed it is unconditional (checked before any figure is
  read).
- Checked `>` vs `>=` operators against the pre-registered bar wording in `V4-RERUN-ROADMAP.md`
  and `ROADMAP.md` Phase 23 SC3 (judge: strict `>`; wp-bench floor: `>=`) — both match.

## Findings

### L-001: v3.0 comparison figures hardcoded as literals, no `source_receipt` field (Low)

**Location:** `scripts/build_eval4_comparison.py` lines 234–238 (`v30_shipping_gen`:
`fresh_full_rerun: 0.4365`, `gate1_reference: 0.4484`) and line 275 (`capture_vs_capture.old:
0.8274`).

These three v3.0-baseline comparison figures are typed directly into the script as Python
literals rather than loaded programmatically from a JSON receipt, and (unlike the four gating
rows checked by Task 2's `--verify` step) they carry no `source_receipt` field in the output
JSON.

**Verified not a correctness bug:** all three values are byte-accurate against their real
sources — `0.4365` and `0.4484` match `output/bench17/wpbench_full_gate_rerun.json`
(`wp_bench_overall` / `gate1_wp_bench_overall`), `0.8274` matches
`output/base21/diagnostic/DIAGNOSTIC_SYNTHESIS.md` (used consistently there as "old-base
capture"). None of these three feed `pre_registered_verdict`'s pass/fail booleans — they are
non-gating reference/comparison context only, which is exactly the scope Task 2's own verify
step (`rows = [candidate_A_raw_base, candidate_B_best_trained, served_s1, capture_ensemble]`)
covers with `source_receipt` presence + `open().close()` checks.

**Impact:** none on the recorded verdict — no gap. Flagged only because "numbers from receipts,
not literals" was a review target: a future edit to the script could silently drift these three
values with no automated check to catch it (the four gating rows would still be caught by
Task 2's verify; these three would not).

**Suggested follow-up (optional, not blocking):** if `build_eval4_comparison.py` is ever
extended, load these three from their source files (or at minimum add
`source_receipt`/`source_receipt_note` fields matching the gating-row convention) rather than
leaving them as bare literals.

## Clean

- CI-aware mechanical criteria: `judge_single_met = ci_lower > 0.85`, `judge_ensemble_met =
  ci_lower > 0.87`, `wpbench_floor = ci_lower >= 0.4286` — all applied on CI-lower bound (never
  point estimate), matching the pre-registered bar operators exactly.
- Fail-closed: `emit_verdict()` unconditionally hard-exits before reading any figure if
  `comparability_audit.json` is absent, `gen_harness_comparable` is false, or
  `needs_confirmatory_gpu_run` is true.
- CI backfill reuses `_bootstrap_ci_lower` imported from `build_gen03_wpbench.py` (not
  reimplemented) — confirmed identical strata/weights/seed to every other arm by independent
  re-run.
- `gen_role_winner_rationale` and the "negative headroom" / "regression-to-teacher" framing in
  `VERDICT-EVAL4.md` are grounded in `output/base21/diagnostic/DIAGNOSTIC_SYNTHESIS.md` (exp5
  regression-to-teacher confirmation, exp1 overtraining, exp4 mix-rebuild result) — not invented
  narrative.
- No debt markers (`TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`) found in the script or the
  three output artifacts.

## Verdict

**status: issues** (1 Low, informational — does not invalidate any recorded number or block the
phase goal).

---
*Reviewed: 2026-07-15*
*Reviewer: Claude (gsd-code-review, standard depth)*
