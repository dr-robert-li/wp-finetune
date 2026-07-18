---
phase: 23-final-evaluation
verified: 2026-07-15T08:12:00+10:00
status: passed
score: 8/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: none
---

# Phase 23: Final Evaluation Verification Report

**Phase Goal:** The new gen+judge pair's actual performance against v3.0's shipping figures is
measured and committed — the milestone's primary verdict.
**Verified:** 2026-07-15T08:12:00+10:00
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `eval4_final_comparison.json` positions BOTH raw base (0.4897) and best-trained ep1 (0.4381) against v3.0 shipping gen (0.4365/0.4484) + 0.4286 floor, and names the gen-role winner | ✓ VERIFIED | `gen_ab.candidate_A_raw_base.overall=0.4897`, `candidate_B_best_trained.overall=0.4381`, `v30_shipping_gen={fresh_full_rerun:0.4365, gate1_reference:0.4484}`, `gen_role_winner="raw_base"` — all present in output/eval4/eval4_final_comparison.json |
| 2 | Pre-registered acceptance criteria (judge rho >0.85 single / >0.87 ensemble; wp-bench >=0.4286) applied mechanically and CI-aware, each recorded | ✓ VERIFIED | `pre_registered_verdict.judge_single_met = ci_lower(0.7125) > 0.85 = False`; `judge_ensemble_met = ci_lower(0.7563) > 0.87 = False`; `wpbench_floor_met_by_gen_role_winner = ci_lower(0.3812) >= 0.4286 = False` — script applies operators on CI-lower, never point estimate; independently re-ran the verify one-liners, matched |
| 3 | Milestone primary verdict + gen-role winner + failure disposition committed under `output/eval4/` before any packaging (27) or conditional-gate (24-26) decision | ✓ VERIFIED | `git log` shows 3 clean commits (b290278, cec5426, c3f636f), `git status` clean on output/eval4/ + script; no `.planning/phases/24-*`..`27-*` directories exist yet (gate/packaging phases not started) |
| 4 | Every candidate figure verified same-harness/stack/seed comparable; sole missing CI (raw base) backfilled offline from its 344-test results file, no new GPU run | ✓ VERIFIED | `comparability_audit.json.gen_harness_comparable=true` (field-equality on 11 fingerprint fields across ep3/ep1/v4b); `needs_confirmatory_gpu_run=false`; independently re-ran `_bootstrap_ci_lower` outside the script on `output/base21/gen03_fresh_new_base_anchor/wp_bench_results_20260714_082330.json` — reproduced `ci_lower=0.3812, ci_upper=0.5983` exactly (deterministic, `bootstrap_seed=1337`) |
| 5 (ROADMAP SC1) | A/B eval runs on wp-bench and judge rho using identical harness (`eval_relabel.py`, vLLM-served, 8192-token cap) | ✓ VERIFIED | `judge_ab.served_s1.methodology="vllm_served"`; `output/base21/judge03_rho.json.vllm_served_single_seed.max_tokens=8192`; wp-bench candidates share the asserted harness fingerprint (seed=1337, n_tests=344, etc.) |
| 6 (ROADMAP SC2) | Results committed to disk before any packaging/gate-continuation decision | ✓ VERIFIED | Same as #3 |
| 7 (ROADMAP SC3) | Pre-registered criteria applied mechanically against measured numbers; met/not-met recorded as milestone's primary verdict | ✓ VERIFIED | `pre_registered_verdict.milestone_primary_verdict` + `disposition="valid_recorded_miss"` present and match the underlying booleans |
| 8 | EVAL4-01 cross-referenced in `.planning/REQUIREMENTS.md` | ✓ VERIFIED | REQUIREMENTS.md line 406: `| EVAL4-01 | Phase 23 | Complete |` |

**Score:** 8/8 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `output/eval4/comparability_audit.json` | Receipt-comparability determination + offline raw-base CI backfill | ✓ VERIFIED | Exists, committed (b290278); all fields populated; independently reproduced the backfilled CI |
| `output/eval4/eval4_final_comparison.json` | Machine-readable EVAL4-01 milestone verdict | ✓ VERIFIED | Exists, committed (cec5426); dual-gen A/B, judge A/B, mechanical verdict block all present |
| `output/eval4/VERDICT-EVAL4.md` | Human-readable milestone verdict narrative | ✓ VERIFIED | Exists, committed (c3f636f); all 5 required sections present, figures read from the JSON (not re-derived — spot-checked, values match exactly) |
| `scripts/build_eval4_comparison.py` | Reusable synthesis script (`--emit audit` / `--emit verdict`) | ✓ VERIFIED | 359 lines, both entry points present and functional (re-ran both verify one-liners against the committed outputs, both pass) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `comparability_audit.json` | `eval4_final_comparison.json` | `emit_verdict()` reads `AUDIT_PATH`, hard-fails (`sys.exit(2)`/`sys.exit(3)`) if absent or if `gen_harness_comparable` is false or `needs_confirmatory_gpu_run` is true | ✓ WIRED | Traced in script lines 156–163; confirmed unconditional (runs before any figure is read) |
| each candidate figure | its source receipt | `source_receipt` path field, opened in Task 2's own verify step | ✓ WIRED | Spot-checked 9 figures byte-for-byte against `output/base21/...` receipts (see 23-REVIEW.md); all match exactly. All `source_receipt` paths resolve on disk. |
| `eval4_final_comparison.json` | `VERDICT-EVAL4.md` | narrative prose reads figures from the JSON, not re-derived | ✓ WIRED | All numbers in VERDICT-EVAL4.md (0.4897, 0.4381, 0.7872, 0.8160, gen_role_winner=RAW BASE, disposition=valid_recorded_miss) cross-checked against eval4_final_comparison.json — exact matches |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Task 1 verify (audit) | `python3 -c "...assert d['gen_harness_comparable'] is True..."` | `audit OK: raw_base_ci_lower= 0.3812` | ✓ PASS |
| Task 2 verify (verdict) | `python3 -c "...assert d['gen_ab']['gen_role_winner']=='raw_base'..."` | `verdict OK: PRIMARY TARGET ... NOT MET ...` | ✓ PASS |
| Task 3 verify (narrative) | `python3 -c "...assert '0.4897' in t and '0.4381' in t..."` | `verdict doc OK 5727 chars` | ✓ PASS |
| Independent bootstrap reproduction | `_bootstrap_ci_lower(raw_base_results_file)` re-run outside the script | `{'ci_lower': 0.3812, 'ci_upper': 0.5983, ...}` (exact match) | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| EVAL4-01 | 23-01-PLAN.md | A/B eval vs v3.0 shipping figures, pre-registered criteria applied, committed before packaging/gates | ✓ SATISFIED | All 3 SCs verified above; REQUIREMENTS.md marks Complete |

No orphaned requirements — REQUIREMENTS.md maps only EVAL4-01 to Phase 23, and it is claimed by 23-01-PLAN.md.

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers in `scripts/build_eval4_comparison.py` or the three `output/eval4/` artifacts. One Low-severity code-quality note from the accompanying code review (23-REVIEW.md, finding L-001): three non-gating v3.0 comparison figures are hardcoded literals without a `source_receipt` field — verified byte-accurate against their real sources, does not invalidate any recorded number, does not block the phase goal.

### Human Verification Required

None. This is a pure JSON/receipt synthesis phase — every claim is mechanically checkable against on-disk receipts, and all checks above were run programmatically (including an independent re-run of the bootstrap CI computation).

### Gaps Summary

None. All 8 must-have truths verified, all 4 artifacts present/substantive/wired, all 3 key links wired, no blocking anti-patterns. The one code-review finding (L-001, Low) does not affect any recorded figure or the milestone verdict.

---

_Verified: 2026-07-15T08:12:00+10:00_
_Verifier: Claude (gsd-verifier)_
