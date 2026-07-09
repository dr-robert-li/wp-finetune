---
phase: 11-compression-packaging
verified: 2026-07-09T20:43:34Z
status: passed_with_notes
score: 8/9 truths verified (1 note — audit-trail accuracy, not a technical failure)
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "The k-sweep audit trail (11-04-SUMMARY FINAL STATE ADDENDUM, SIEVE-DECISIONS.md, optimal_k.json, prune_set_for_phase13.json) accurately records what was executed and why k=13/k=32-judge lack rho scores"
    status: partial
    reason: >
      Disk evidence (logs/sieve/ksweep_driver_resume.log, output/sieve/ksweep/{gen,judge}_k{32,13}/*)
      contradicts the documented narrative. 11-04-SUMMARY's FINAL STATE ADDENDUM (committed
      fcade46, 2026-07-10 02:44:13) claims "ran k=64 fully and k=32 gen before the executor
      session died mid-k=32-judge" and that k=13 was "deliberately NOT run." The driver log shows
      the opposite: judge_k32/{s0,s1,s2} all completed by 23:42 on 2026-07-09 (BEFORE the addendum
      was even written), and the sweep continued past the addendum's write time to run k=13 gen
      (timed out after 7200s, a real measured failure, not a skip) and k=13 judge captures for all
      3 seeds (121/121 each, completed by 04:18 on 2026-07-10), ending with an explicit
      "=== k-sweep COMPLETE ===" log line. optimal_k.json (committed 02:50, 6 min after the
      addendum) and prune_set_for_phase13.json / SIEVE-DECISIONS.md (committed 06:35, ~2h AFTER
      the driver actually finished at 04:18) both still repeat the stale "never
      executed"/"session died" framing without re-checking the by-then-available real data.
    artifacts:
      - path: ".planning/phases/11-compression-packaging/11-04-SUMMARY.md"
        issue: "FINAL STATE ADDENDUM claims a mid-run session death and 'k=13 deliberately NOT run'; driver log shows the sweep ran to completion including k=13."
      - path: "output/sieve/optimal_k.json"
        issue: "k=13 recorded measured:false with note 'Never executed'; k=32 judge_rho_note says 'sweep session ended before judge capture' — both false per logs/sieve/ksweep_driver_resume.log and the completed judge_k13/judge_k32 capture directories."
      - path: ".planning/phases/11-compression-packaging/SIEVE-DECISIONS.md"
        issue: "SIEVE-05 verdict table repeats 'k=13 | not run' — should read 'executed, unparseable' per the corrected finding below."
    missing:
      - "Correct the addendum/optimal_k.json/SIEVE-DECISIONS.md/prune_set_for_phase13.json to state: k=13 gen timed out (7200s, real failure); k=13 AND k=32 judge captures completed for all 3 seeds (121/121) but 0/121 responses were parseable into rubric dimension scores at either k (verified independently during this verification via eval.output_parsers.parse_judge_scores over both capture sets) — i.e. total judge-output collapse under aggressive masking, not a process failure. This is a stronger, not weaker, confirmation of the monotone-collapse finding and does not change optimal_k=full, but the audit trail must reflect what actually happened for a decision record whose entire purpose is auditability."
      - "Re-affirm (or re-issue) the human sign-off with the corrected narrative, even though the substantive decision is expected to be unchanged, since the sign-off was given a table describing a session death that (per the logs) had already been superseded by real — if degenerate — completed data at the time later documents were committed."
deferred: []
---

# Phase 11: Compression & Packaging (Training-Free Sieve) Verification Report

**Phase Goal:** Training-free MoE-Sieve for the shipped two-model pair — routing profiles → inference-time
expert-masking k-sweep → TOST optimal-k declaration with protected experts retained → prune-set hand-off
to Phase 13. No training, no new weights.

**Verified:** 2026-07-09T20:43:34Z (session clock; project calendar 2026-07-10)
**Status:** passed_with_notes
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Protected expert mask (1,480 experts) is inviolable and untouched by Phase 11 | VERIFIED | `protected_expert_mask.npy` sha256 `659af6eb…` unchanged since Phase 7 (mtime 2026-06-15); shape (48,128) bool, sum 1480, re-verified directly by this verifier. `.json` sha256 `ade549e0…` matches every recorded checksum across 11-03/11-04/11-05 summaries and `prune_set_for_phase13.json` |
| 2 | No training / no new weights anywhere in the phase | VERIFIED | All new scripts (`sieve_expert_mask_inference.py`, `sieve_ksweep_run.py`, `sieve_capture_judge_http.py`, `tost_gate.py`, `sieve_cross_seed_overlap.py`, `sieve_protected_retention.py`) grepped for optimizer/backward/gradient calls — only docstring mentions ("No training, no gradients"), zero training code. The s0/s2 "merges" (11-02) are export+merge of pre-existing Phase-10 Tinker checkpoints into servable shards, not new training |
| 3 | GB10 sequential serving honored (no co-resident model instances) | VERIFIED | `logs/sieve/ksweep_driver_resume.log`: every arm shows `[vllm] booting ... ` immediately followed by `[vllm] stopped ...` before the next model boots, for gen and all 3 judge seeds, every k |
| 4 | `layer_stability_notes` carried into `prune_set_for_phase13.json` verbatim | VERIFIED | Byte-for-byte identical dict (`added`, `low_jaccard_band`, `note`, `late_layer_shift`, `mask_immutable` keys) between `protected_expert_mask.json` and `prune_set_for_phase13.json`, diffed directly by this verifier |
| 5 | SIEVE-01: fresh routing profiles of 3 judge seeds + protected-subset verification | VERIFIED | `output/sieve/judge-s{0,1,2}/routing_report.jsonl` (48/48/48 records each, per 11-03-SUMMARY); `sieve_protected_retention.py` confirms mask subset of shared top-64 (0 at risk), 866/198 at-risk at k=13/32 correctly flagged for forced retention |
| 6 | SIEVE-02: explicit N/A documentation under training-free scope | VERIFIED | `SIEVE-DECISIONS.md` documents the two supersession events (RL rejection 2026-07-05, training-free lock 2026-07-08) with a clear rationale; not silent |
| 7 | SIEVE-03: 30/70 ratio traceability to Phase 4/7 | VERIFIED | `SIEVE-DECISIONS.md` traces the shipped ratio to `qwen3-30b-wp-30_70-reasoning-merged-v4` (Phase 4) and the Phase 7 matched stimulus (`ratio_30_70`); no new ratio decision made, consistent with all k-sweep artifacts |
| 8 | SIEVE-04: k-sweep produces decision-grade evidence at 3+ budgets with protected retention at every measured k | VERIFIED (see note) | `k_sweep_results.json`: full/64/32 measured with real wp-bench + judge-rho numbers, monotone catastrophic collapse (0.4484→0.2275→0.0546 wp-bench; 0.8075→0.5415→unparseable judge); k=13 gen genuinely attempted (7200s timeout) and k=13/k=32 judge captures completed for all 3 seeds (121/121) but scored 0/121 parseable by `eval.output_parsers.parse_judge_scores` (re-verified independently in this pass) — protected_retained=true at every measured k. **Note:** the documentation of *why* k=13/k=32-judge lack a rho number is materially inaccurate — see Gaps below |
| 9 | SIEVE-05: TOST gate declares optimal k with protected retention + judge-rho bar, human-approved, prune-set handed to Phase 13 | VERIFIED | `scripts/tost_gate.py` runs two one-sided Welch t-tests (statsmodels absent per `sieve_env_precheck`, hand-rolled per plan); `optimal_k.json` records `optimal_k="full"`, `no_equivalent_k=true`, per-k TOST math independently re-derivable from the recorded mean_diff/CI; `human_signoff` block present (Dr. Robert Li, 2026-07-10, AskUserQuestion); `prune_set_for_phase13.json` is the single, complete Phase-13 hand-off (optimal_k, 1,480 protected sha-pinned, hot/cold per layer, layer_stability_notes, regression bars) |

**Score:** 8/9 truths cleanly verified; 1 truth (#8) verified on substance but flagged for a documentation-accuracy gap that must be corrected in the audit trail (see Gaps).

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/sieve_env_precheck.py` | disk/mem/statsmodels gates | VERIFIED | Present, self-check asserts pass |
| `tests/test_sieve_protected_retention.py`, `test_tost_gate.py`, `test_sieve_ksweep_mask.py`, `test_sieve_cross_seed_overlap.py` | Wave-0 contracts, GREEN by end of phase | VERIFIED | 15 passed, 0 skipped, 0 errors (re-ran directly: `.venv-tinker/bin/python -m pytest tests/test_sieve_protected_retention.py tests/test_tost_gate.py tests/test_sieve_ksweep_mask.py tests/test_sieve_cross_seed_overlap.py -q`) |
| `models/_staging/qwen3-30b-wp-v1.3-{s0,s2}-merged/` | 13-shard bf16 judge seeds | VERIFIED (per 11-02-SUMMARY; gitignored, not independently re-hashed this pass — merge-convention pytest suite (7 passed) already certified per-expert byte-exact extraction) | staging dirs present per 11-02 self-check |
| `scripts/sieve_cross_seed_overlap.py`, `scripts/sieve_protected_retention.py` | cross-seed Jaccard + mask subset check | VERIFIED | `output/sieve/cross_seed_overlap.json` (mean_overlap 0.9332 ≥ 0.90 → shared profile), `output/sieve/protected_retention_check.json` (0/198/866 at-risk at k=64/32/13) both present and internally consistent |
| `scripts/sieve_expert_mask_inference.py`, `scripts/_sieve_vllm_patch/sitecustomize.py`, `scripts/sieve_capture_judge_http.py`, `scripts/sieve_ksweep_run.py` | inference-time masking + k-sweep driver | VERIFIED | All present; driver log confirms live use through 4 arms |
| `output/sieve/k_sweep_results.json` | full+64+32+13 arms | VERIFIED — but see note above | All 4 arms present in `sweep[]`; `halted: false` at file level (the sweep completed; earlier halt was on an *aborted first attempt*, `logs/sieve/ksweep_driver.log`, superseded by the resumed run) |
| `scripts/tost_gate.py`, `output/sieve/optimal_k.json` | TOST equivalence + optimal-k | VERIFIED | Math re-derivable from recorded CI/p-values; `no_equivalent_k=true` correctly follows from k=64/32 TOST rejections |
| `output/sieve/prune_set_for_phase13.json` | Phase-13 hand-off | VERIFIED | `optimal_k`, `protected_experts.count=1480` (sha-pinned, matches live mask), `layer_stability_notes` (verbatim), `hot_cold_per_layer` (all-keep, 48 layers), `sieve_profile_mode="shared"`, regression bars all present |
| `.planning/phases/11-compression-packaging/SIEVE-DECISIONS.md` | SIEVE-02/03 + reinterpretation record | VERIFIED (content accurate for SIEVE-02/03/reinterpretation; the SIEVE-05 k=13 row needs the correction noted above) | Present, substantively documents the training-free scope, only the k=13 execution-status line needs correction |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| k-sweep results → optimal-k decision | `output/sieve/k_sweep_results.json` | `scripts/tost_gate.py` reads per-item jsonl arrays from `output/sieve/ksweep/gen_k*/wp_bench_results_*.jsonl` | WIRED | Confirmed real per-item data consumed (344 tests/arm), not just aggregates |
| Phase 11 → Phase 13 hand-off | `output/sieve/prune_set_for_phase13.json` | single consumed artifact, all required keys present | WIRED | `optimal_k`, `protected_experts`, `layer_stability_notes`, `hot_cold_per_layer` all present and internally consistent with upstream artifacts |
| Human sign-off → prune-set emission | `optimal_k.json.human_signoff` → `prune_set_for_phase13.json.human_signoff` | ordering enforced (T-11-13) | WIRED, content accurate for the *decision*, inaccurate for the *evidence table presented* | Sign-off itself (lock optimal_k=full) is well-supported by the k=64/k=32 wp-bench collapse alone (22pp / 39pp gaps, TOST p≈1e-12/1e-40) independent of the k=13/k32-judge documentation issue — the decision does not appear to depend on the inaccurate framing, but the reviewer was shown it nonetheless |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Wave-0 pytest contracts pass | `.venv-tinker/bin/python -m pytest tests/test_sieve_protected_retention.py tests/test_tost_gate.py tests/test_sieve_ksweep_mask.py tests/test_sieve_cross_seed_overlap.py -q` | `15 passed` | PASS |
| Protected mask integrity | `sha256sum` + `numpy.load` on `protected_expert_mask.{npy,json}` | sha256 matches all recorded values; shape (48,128) bool sum=1480 | PASS |
| k=13/k=32 judge captures independently re-scored | `eval.output_parsers.parse_judge_scores` over all 6 capture files (k13×3 seeds, k32×3 seeds) | 0/121 parseable in every one of the 6 files | PASS (confirms total collapse, not a scoring bug) |
| `layer_stability_notes` verbatim carry | direct dict comparison, mask JSON vs prune-set JSON | byte-identical | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SIEVE-01 | 11-02/11-03 | Fresh routing profiling; protected retained | SATISFIED (reinterpreted) | Training-free reinterpretation locked in 11-CONTEXT.md/SIEVE-DECISIONS.md: no LoRA applied (RL rejected, no training path); 3 judge-seed profiles + protected-subset verify delivered instead. **Note:** REQUIREMENTS.md's SIEVE-01 checkbox text still reads literally as "LoRA r=32... applied to hot routed experts... cold routed experts frozen" on an "RL-trained model" — none of that occurred. The `[x]` is correct under the locked reinterpretation but REQUIREMENTS.md itself carries no annotation of the reinterpretation (only 11-CONTEXT.md/SIEVE-DECISIONS.md do) |
| SIEVE-02 | 11-05 | Data-routing spec | SATISFIED (N/A, documented) | See SIEVE-DECISIONS.md |
| SIEVE-03 | 11-05 | 30/70 ratio traceability | SATISFIED (traceability only) | See SIEVE-DECISIONS.md |
| SIEVE-04 | 11-04/11-05 | K-sweep ≥3 budgets | SATISFIED | See Truth #8 and Gaps |
| SIEVE-05 | 11-05 | TOST optimal-k + protected retention | SATISFIED | See Truth #9 |

No orphaned SIEVE requirements found in REQUIREMENTS.md's phase-mapping table (lines ~343-347); all five map to Phase 11 and are checked.

### Anti-Patterns Found

None of TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER found in the phase's new scripts (`sieve_*.py`, `tost_gate.py`). No stub `return null`/empty-array patterns found feeding rendered output (this is a batch-analysis phase, not UI).

### Gaps Summary

**One substantive-but-non-blocking gap: audit-trail accuracy for the k=13 / k=32-judge non-measurement.**

The documented narrative across `11-04-SUMMARY.md`'s FINAL STATE ADDENDUM, `output/sieve/optimal_k.json`, and `SIEVE-DECISIONS.md` says the k-sweep session "died" mid-k=32-judge and that k=13 was "deliberately NOT run." This verifier found, via `logs/sieve/ksweep_driver_resume.log` and the actually-completed capture files under `output/sieve/ksweep/{judge_k13,judge_k32}/{s0,s1,s2}/judge_responses.jsonl`, that:

- The addendum (committed 2026-07-10 02:44:13) was written *after* k=32's judge captures had already fully completed (23:42 on 2026-07-09) and *before* the still-running background driver went on to execute k=13 (gen timed out at 7200s — a real, measured failure; judge captured 121/121 for all 3 seeds, finishing 04:18 on 2026-07-10, ending in an explicit `=== k-sweep COMPLETE ===` log line).
- `optimal_k.json` (committed 6 minutes after the addendum) and `prune_set_for_phase13.json`/`SIEVE-DECISIONS.md` (committed ~2 hours *after* the driver had actually finished) all still repeat the stale, by-then-superseded narrative, without checking the newer data that was sitting on disk.
- Independently re-scoring the real captured k=13/k=32 judge responses (this verifier ran `parse_judge_scores` over all 6 files) confirms **0/121 parseable** in every case — the judge model's output under aggressive masking degenerates into unparseable repeat-loop garbage (e.g. `` ```php```php... `` or `$ $ $ $...`), which is *why* `judge_ensemble_rho` came back `null` in the driver's own scoring function, not because the arm was skipped.

**This does not change the SIEVE-05 decision.** `optimal_k="full"` is independently and solidly supported by the k=64/k=32 wp-bench TOST rejections alone (22pp and 39pp below full, p≈1e-12/1e-40) — the corrected k=13/k=32-judge picture (total output collapse) is if anything *stronger* evidence for the same conclusion, not contradictory. The human sign-off's substantive lock is expected to be unaffected. But the artifacts that exist specifically to make this decision auditable contain a materially false description of what was executed, and that must be corrected: replace "never executed"/"session died" language in `11-04-SUMMARY.md`, `optimal_k.json`, `prune_set_for_phase13.json`, and `SIEVE-DECISIONS.md` with the accurate finding (executed; gen timeout at k=13; total judge-output collapse, 0/121 parseable at both k=13 and k=32).

**Secondary, informational note (not a gap):** REQUIREMENTS.md's SIEVE-01 checkbox text is the pre-reinterpretation literal spec (LoRA training on an RL-trained model) and carries no in-file annotation of the training-free reinterpretation; the reinterpretation is fully documented elsewhere (11-CONTEXT.md, SIEVE-DECISIONS.md) and was a locked, user-approved 2026-07-08 scope decision, so this is reported for traceability only, not as a defect.

---

_Verified: 2026-07-09T20:43:34Z_
_Verifier: Claude (gsd-verifier)_
