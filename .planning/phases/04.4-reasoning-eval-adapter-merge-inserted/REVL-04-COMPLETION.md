# REVL-04 wp-bench HARD GATE — Completed 2026-05-30

## Result: PASS (real, fair, non-degenerate)

| Model | overall | knowledge | correctness (execution) |
|-------|---------|-----------|-------------------------|
| baseline (merged-v2) | **0.4286** | 0.500 (160/320) | 0.375 |
| reasoning-merged | **0.4616** | 0.494 (158/320) | **0.4375** |

`meets_baseline: true` — reasoning ≥ baseline. Reasoning trades a hair of
knowledge accuracy for higher execution-code correctness. Full 344-test suite
(320 knowledge + 24 execution) per model. Both result files contain **0** rows
with leftover `<think>` scaffold → strip applied symmetrically, fair comparison.

Artifacts:
- `output/04.4_wp_bench_results.json` (`pass: true`, `meets_baseline: true`)
- `output/eval_reasoning/summary.json` (REVL-01/02/04 gates)
- `output/eval_reasoning/{baseline_30_70,reasoning_merged}/wp_bench_results_*.json`

## All three cascade HARD gates now PASS

- REVL-02 PHPCS: 1.0 == 1.0
- REVL-01A Spearman: 0.350 ≥ 0.171 (relative gate per PLAN lines 105/461/808; the
  absolute 0.85 was a Phase-4-*triage* threshold, superseded by Option-3 redesign)
- REVL-04 wp-bench: 0.4616 ≥ 0.4286

## Why REVL-04 was failing (fix chain — all in `scripts/run_eval_reasoning.py`)

wp-bench had **never** run successfully (always "manual-pending"). Eight distinct
blockers, each surfaced by staged single-failure observation:

1. **wp_env_dir path** — `HarnessConfig.from_file` resolves relatives against the
   *config file's dir*; orchestrator wrote tmp config into `output/.../<tag>/`, so
   `./wp-bench/runtime` → missing. Fix: write absolute `wp_env_dir`.
2. **wp-env not installed** — submodule root-owned, `npm install` blocked. Fix:
   `npm install -g @wordpress/env` (user prefix); wp-env stores instance in `~/.wp-env`,
   only reads (readable) runtime dir.
3. **npx registry flakiness** — `wp-env` is a global bin, not a registry package;
   bare `npx wp-env` 404s/hangs. Fix: `scripts/_wpbench_shim/npx` execs directly.
4. **node IPv6 happy-eyeballs** — wp-env's config-read GitHub call intermittently
   ETIMEDOUTs. Fix: `NODE_OPTIONS=--dns-result-order=ipv4first`.
5. **litellm provider prefix** — model name `wp-30_70` → "LLM Provider NOT provided".
   Fix: `openai/wp-30_70`.
6. **litellm api_base ignored** — wp-bench `models.py:25` calls `litellm.completion()`
   WITHOUT api_base/api_key; hit real OpenAI. Fix: `OPENAI_API_BASE`/`OPENAI_BASE_URL` env.
7. **litellm api_key** — Fix: `OPENAI_API_KEY=EMPTY` (vLLM ignores value).
8. **`<think>` scaffold breaks knowledge scoring** — Qwen3 chat template emits empty
   `<think>\n\n</think>` prefix; `core.py:134` `answer.startswith(correct_answer)` fails
   for BOTH models. Fix: `scripts/_wpbench_pth/usercustomize.py` runtime-patches
   `ModelInterface.generate` to strip `<think>` (mirrors `eval_gen.py:60`), symmetric.
   Plus score-extraction bug: wp-bench writes timestamped `wp_bench_results_<ts>.json`,
   score at `metadata.scores.overall` (not configured path / `score` key).

## Outstanding for full Phase 4.4 close (NOT done here)

The W2-02 cascade covers REVL-01/02/04 only. Still outstanding per PLAN:
- REVL-03 (Claude-evaluator agent quality)
- REVL-05 (**human review / sign-off** — required before merge per PROJECT decision)
- REVL-06 (fix correctness aggregate)
- REVL-07 (classification matrix — SOFT/flag)
- REVL-08 (reasoning length — SOFT/flag)

Merge to `models/qwen3-30b-wp-30_70-reasoning-merged/` already exists (smoke-certified).
Human sign-off + remaining gates are a human decision — not recorded by automation.

Note: PLAN specified `tests/phase4_4/test_wp_bench_artifact.py` and `test_promote_gated.py`
as the artifact contract, but those files were never scaffolded (only test_output_parsers,
test_revl01_helpers, test_smoke_common exist). Artifact shape verified by inspection.

## Prereqs to reproduce (host state NOT captured in repo)

Orchestrator changes are committed/portable, but a fresh run also needs:
1. `npm install -g @wordpress/env`  (global wp-env binary on PATH)
2. `cd wp-bench/runtime && wp-env start`  (provisions `~/.wp-env` WordPress + Docker
   containers; ~2-3 min first run; needs Docker daemon + network)
   - If site breaks (`wp core is-installed` fails): `wp-env destroy && wp-env start`
3. The npx shim (`scripts/_wpbench_shim`), IPv4 NODE_OPTIONS, OPENAI_* env, and the
   `<think>`-strip PYTHONPATH are all baked into `_run_wpbench` — no manual setup.

Then: `python -m scripts.run_eval_reasoning --wpbench-only` (reuses prior eval numbers
from summary.json; boots vLLM per model). Add `--skip-wpbench` to skip REVL-04.
