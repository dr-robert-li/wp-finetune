---
phase: 22-sieve-protected-mask-tooling-adaptation
reviewed: 2026-07-15T16:30:00+10:00
status: pass
scope: "git diff 33b8615^..8e7fe00 -- scripts/ (9 files, +748/-103)"
files_reviewed:
  - scripts/sieve_arch.py
  - scripts/sieve_v4_tooling_smoke.py
  - scripts/profile_base_model.py
  - scripts/profile_merged_model.py
  - scripts/extract_protected_mask.py
  - scripts/sieve_expert_mask_inference.py
  - scripts/sieve_cross_seed_overlap.py
  - scripts/sieve_protected_retention.py
  - scripts/_sieve_vllm_patch/sitecustomize.py
findings:
  critical: 0
  blocker: 0
  warning: 0
  info: 2
---

# Phase 22 Code Review — Sieve/Protected-Mask Tooling Adaptation

**Scope:** the 5 commits `33b8615..8e7fe00` touching `scripts/` — `sieve_arch.py` (new), the wiring of 6
consumer scripts, the vLLM patch class-resolution refactor, and the new `sieve_v4_tooling_smoke.py`.

## Summary

Mechanical, well-scoped parameterization. Every hardcoded v3 dimension (`48`, `128`, `1480`) that was
load-bearing is now derived from `model.config` (via `sieve_arch.arch_dims`/`layer_strata`) or from the
profiling JSONL itself (via `sieve_arch.infer_dims_from_records`). Fail-loud discipline is real, not
cosmetic, in both places it matters. No debt markers (`TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`) found
in any reviewed file. No blockers, no criticals.

## Fail-loud resolution — verified, not cosmetic

- `sieve_arch.resolve_moe_layers` (scripts/sieve_arch.py:93-127): walks the ordered candidate-root tuple,
  returns as soon as a root yields a non-empty layer list, and **raises** `RuntimeError` naming every tried
  path if none resolve. No `except: pass`, no default-to-empty-list return. Confirmed by
  `tests/test_sieve_arch.py` (`resolve_moe_layers` raises on an `EmptyModel()` stub) and by the module's own
  `demo()` self-check (`scripts/sieve_arch.py:214-218`).
- Both profilers additionally assert `len(hooks) == n_layers` after calling `resolve_moe_layers`
  (`scripts/profile_base_model.py:474`, `scripts/profile_merged_model.py:176`) — a second, independent
  fail-loud gate even if a future traversal-root change ever yielded a *partial* (non-zero, non-40) count.
- `_resolve_moe_block_class` (scripts/_sieve_vllm_patch/sitecustomize.py:47-72): tolerates `ImportError` per
  candidate (tries the next), but raises `RuntimeError` naming every tried candidate once the list is
  exhausted. The `_install()` caller does not swallow that exception — it re-raises after logging
  (`sitecustomize.py:150-153`), so a `SIEVE_KEEP_MASK_NPY`-set-but-unresolvable-class run crashes container
  boot rather than silently serving unmasked. Confirmed by `tests/test_sieve_vllm_patch.py::test_raises_when_no_candidate_resolves`.
- The ambiguous-scan case (2+ `*SparseMoeBlock` classes in one resolved module) correctly does **not** guess
  — falls through to the next candidate / raises, rather than picking arbitrarily
  (`test_ambiguous_scan_does_not_resolve`).

## No remaining load-bearing 48/128/1480 literal

Grep sweep of all 9 touched files for `\b(48|128|1480)\b` found only:
- `scripts/sieve_cross_seed_overlap.py:41-42`, `scripts/sieve_expert_mask_inference.py:54-55` — module
  constants `N_LAYERS=48`/`N_EXPERTS=128` kept for import back-compat, explicitly commented "NOT load-bearing"
  and confirmed unread by any function body (`load_seed_counts` now infers dims from the file itself; falls
  back to these constants only for a genuinely empty input file — a degenerate case where any placeholder
  value is equally correct).
- `scripts/profile_base_model.py:118-119` — `RoutingCollector.__init__` keyword defaults (`n_layers=48,
  n_experts=128`); every real call site (`profile_base_model.py:461`, `profile_merged_model.py:164`) passes
  explicit `n_layers=n_layers, n_experts=n_experts` derived from `sieve_arch.arch_dims(model.config)` —
  verified by reading both call sites, not just grep.
- `scripts/profile_base_model.py:71` — `compute_eeff(expert_counts, n_experts=128)`: `n_experts` is not read
  anywhere in the function body (docstring says "used for output range context only"); pre-existing dead
  parameter, not introduced by this phase.
- All other matches are docstrings/comments/module docstrings describing v3 for contrast, or the `demo()`
  self-check's own v3 fixture literals (`sieve_arch.py:174,188` — intentionally hardcoded v3 fixture values,
  correct usage).

No hit is load-bearing in an actual v4 code path.

## v3 back-compat

`tests/test_sieve_arch.py`, `test_protected_mask.py`, `test_sieve_cross_seed_overlap.py`,
`test_sieve_ksweep_mask.py`, `test_sieve_protected_retention.py` all carry both the original `(48,128)`/`1480`
v3 fixture cases AND new `(40,256)` v4 cases — confirmed by reading the diffs (additive, no v3 case removed
except `sieve_protected_retention.py`'s literal `shape==(48,128)`/`sum==1480` hard-asserts, which the plan
explicitly scoped for removal since the v4 mask is a fresh Phase-25 profile of unknown count — replaced with
`dtype==bool` + non-empty, both still true for v3). 47/47 tests pass
(`pytest tests/test_sieve_arch.py tests/test_protected_mask.py tests/test_sieve_cross_seed_overlap.py
tests/test_sieve_ksweep_mask.py tests/test_sieve_protected_retention.py tests/test_sieve_vllm_patch.py -q`).

## AutoModelForImageTextToText loader fix — correctness

`scripts/sieve_v4_tooling_smoke.py:109-114` loads the v4 judge checkpoint via
`AutoModelForImageTextToText.from_pretrained(...)` instead of the plan's literal `AutoModelForCausalLM`. The
SUMMARY documents the reasoning (a meta-device state_dict key-set diff: `Qwen3_5MoeForCausalLM` has 692/693
keys missing against this checkpoint's nested `model.language_model.layers.*` convention;
`Qwen3_5MoeForConditionalGeneration`, resolved by `AutoModelForImageTextToText`, has 0 missing keys). This is
architecturally sound — the checkpoint's `config.json` `architectures` field
(`Qwen3_5MoeForConditionalGeneration`) and the VL-composite `text_config`/`vision_config` split corroborate a
composite checkpoint, and `AutoModelForCausalLM` resolving to the flat-text class would indeed silently build
a randomly-initialized text backbone that still passes every shape-based assert (a genuine spoofing-adjacent
failure mode the fix correctly closes). The receipt's real evidence (4m12s load, 1026/1026 shards, non-mock
`router_logits_last_dim: 256` captured from an actual forward-hook fire) is consistent with a real weight
load, not a fabricated result. Not independently re-run in this review (would consume a second bounded GB10
load); accepted on the strength of the documented key-diff methodology + receipt internal consistency.

## Mask-inference strata math

`scripts/profile_merged_model.py:245-268`: `strata_eeff` is built from `full_eeffs_total` — captured via
`collector.get_layer_eeffs(layer_idx)[0]` (index 0 is confirmed `eeff_total`,
`scripts/profile_base_model.py:244-250`) — **before** the Jaccard subsample pass, which the code comment
correctly flags mutates collector state in place. Grouping by `sieve_arch.layer_strata(model.config)` and
filtering NaN before `np.array(...).mean()/.max()/.var()` is correct (a stratum with zero valid layers
degrades to `{mean: None, max: None, var: None, n_layers: 0}` rather than raising or silently emitting `NaN`
into a downstream JSON). Added to the return dict and the `jaccard_stability.json` sidecar as documented,
alongside (not replacing) the existing collapsed stats.

## Findings

### Info (2, non-blocking)

**I-1: `RoutingCollector.__init__` keyword defaults still read `n_layers=48, n_experts=128`.**
Not load-bearing (every real call site overrides them), but a future caller that forgets to pass explicit
dims would silently default back to v3 shape. Low risk given the current call-site discipline; worth a
follow-up note if a new consumer of `RoutingCollector` is ever added outside the two audited profilers.

**I-2: ROADMAP Phase 22 SC1 literal text ("`model.model.language_model.layers`") does not match the actual
resolved traversal root.** 22-VALIDATION.md explicitly reconciles this (Phase 20-04's empirical flat-tree
finding overrides the ROADMAP's literal guess) and the 22-02 receipt records the true empirical root
(`model.language_model.layers`) — this is documented, intentional, and not a code defect. Flagged here only
so it's visible to whoever reads SC1 literally without the VALIDATION.md context. See VERIFICATION.md for the
disposition.

## Verdict

**PASS.** No fixes applied — nothing found rose to Critical/Blocker severity.

---
*Reviewer: Claude (gsd-code-review)*
*Reviewed: 2026-07-15*
