# Phase 16-01 — Summary

**Completed:** 2026-07-10
**Requirements:** PIPE-01, PIPE-02, PIPE-03

## What shipped

- **`PIPELINE.md`** (PIPE-01) — the frozen end-to-end method. Spine (data -> SFT gen + judge -> merge ->
  eval) plus three conditional compression gates (RL rejected, Sieve full, prune no_winner), each with its
  runnable entrypoint, gate, and known Qwen3-30B-A3B result. No-winner gates kept as conditional re-test
  stages for the next base. All referenced entrypoints verified present on disk.
- **`deprecated/`** (PIPE-02) — 95 one-off experiment scripts moved out of `scripts/` with a README.
- **Repo cleanup** (PIPE-03) — `logs/` added to `.gitignore`; root stray `judge_batch_18.py` deprecated;
  README points at PIPELINE.md and deprecated/.

## Deprecation — how it was made safe

Rule: underscore-prefixed = one-off, out of pipeline. But the map was verified against reality, not
trusted. A background repo-map agent grepped candidates against the active tree; I then re-verified every
Python-importable candidate and caught **three real active dependencies the map had misclassified as dead**:

- `_rlev01_wpbench_ckpt.py` — imported by the live `rl_codegen_tripwire.py` (single source of truth for the
  codegen bar `BASELINE_V12`). **Kept in scripts/.**
- `_p0_vllm_smoke_serve.py` — the vLLM boot/health/stop helper imported by `run_eval_reasoning.py`,
  `sieve_ksweep_run.py`, `prune_gated_eval.py`, `capture_reasoning_responses.py`. **Restored to scripts/.**
- `_reward_validity_oracle.py` — imported by the active `reward_validity_gate.py`. **Restored to scripts/.**

`reward_form_sweep.py` (dead middle-tier) was moved together with its helper `_probe_rl_reward.py` so the
import stays resolvable inside `deprecated/`. `scripts/relabel/` was NOT moved — it holds the active judge
relabel-eval entrypoint (`eval_relabel.py`) referenced by PIPELINE.md.

**Verification:** after the moves, `grep` finds zero active imports of any moved module, and
`python3 -m py_compile scripts/*.py eval/*.py` compiles clean.

## Deviations

- **Middle-tier scripts left in place.** The map flagged ~40 non-underscore scripts with no automated
  caller (build_*, audit_*, revl0*, generate_cot variants). These are plausible human-run gates or stage
  variants; adjudicating each is over-reach and risks breaking a manual workflow. The safe, high-value
  cut was the 95 verified-dead one-offs. Left as a follow-up if a deeper prune is wanted.
- **Model dirs not moved.** Too large; superseded merges/tombstones stay under `models/` (gitignored) as
  evidence.

## Self-check

- `PIPELINE.md`, `deprecated/README.md` on disk: yes.
- Active tree imports of moved modules: 0 (grep).
- `py_compile scripts/*.py eval/*.py`: clean.
- 95 files in `deprecated/scripts/`; 3 active underscore deps retained in `scripts/`.
