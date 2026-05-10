# Phase 0 — Diagnostic Baseline

Status (2026-05-11).

| Step | Description | Status | Output |
|------|-------------|--------|--------|
| 0.1  | `profile_base_model.py` on base + 30/70 adapter — E_eff(gen) vs E_eff(judge) | **PENDING** — GPU container required | `output/diagnostic/profiling_{base,30_70}/` |
| 0.2  | `rubric_scorer.py` on 27 human + 93 UGC + 25 boundary seeds | **DONE** (with caveats — see below) | `output/diagnostic/seed_scorer_agreement.{json,md}` |
| 0.3  | `eval_judge.py` on 30/70 adapter vs seed-derived GT — Spearman | **READY** — synthesizer built, GPU container required | `output/diagnostic/judge_30_70_seed_spearman.json` |

## Step 0.2 — final results (full 4-tool, snippet auto-wrap, relaxed N/A, corrected dim mapping)

145 / 145 seeds scored. Distribution: **min 64.6, max 100, mean 95.8, stdev 9.7**.

Per-dimension Spearman vs human dim scores:

| Dim | n | Spearman | p | Notes |
|-----|---|----------|---|-------|
| D1_wpcs | 22 | +0.087 | 0.702 | weak |
| D2_security | 15 | +0.175 | 0.532 | weak |
| D3_sql | 12 | **+0.996** | 0.000 | **TRIVIAL — bimodal**: 10 pairs at (h=2, r=2), 2 pairs at (h=8, r≈10). Not evidence of subtle ranking. |
| D4_perf | 8 | +0.286 | 0.493 | weak |
| D5_wp_api | 31 | **−0.345** | 0.057 | **NEGATIVE** — scorer ranks WP-API quality opposite of humans. Real concern. |
| D6_i18n | 16 | nan | nan | rubric output constant 10 (no checks fire) |
| D7_a11y | 12 | nan | nan | rubric output constant 10 |
| D8_errors | 8 | nan | nan | rubric output constant 10 |
| D9_structure | 15 | −0.071 | 0.800 | weak |

### Honest interpretation

1. **Original plan gate (Pearson ≥ 0.75) fails at every dimension as written.** D3_sql looked passing but is bimodal trivial. D5_wp_api is actively reversed.
2. **Rubric scorer is a defect detector, not a defect ranker.** PHPCS / PHPStan static checks fire on known-bad patterns, but the gradations humans assign (a 2/10 vs a 4/10) reflect semantic severity that static tools cannot extract. The 18 LLM-assisted checks (rubric §F.5) were designed for exactly this — they are still deferred (`_LLM_CHECK_COUNT = 18` in `rubric_scorer.py`).
3. **D5_wp_api −0.345 is the biggest red flag.** With LLM checks off, the only fires are deterministic anti-patterns (e.g. missing nonce, wrong WP API). Humans down-score plugins for things the scorer can't see (capability checks, sanitisation gaps). So rubric tends to grade defective code as 10 and lightly-defective code as 7–9, inverting the human ordering.
4. **Structural improvements over Phase 0 run #1:** distribution is unimodal not bimodal; no all-N/A blanking; full 4-tool active. Architecture is sound; calibration evidence is not yet sufficient to use seeds as the *only* anchor.

### Scorer change-set this session

`eval/rubric_scorer.py` and `eval/rubric_definitions.py` were modified during Phase 0:
- `score_code` now auto-wraps bare snippets with `<?php` prefix (was: PHPCS returned `_unavailable: True` for 33/145 seeds).
- `NA_DETECTION_HINTS` relaxed: D1/D5/D9/D8 now match function-body forms; D2 broader; D6/D7 unchanged.
- **Prior triage results (`output/triage_decision.md`, Spearman 0.5698 for 30/70) are not apples-to-apples with anything run after 2026-05-11.** STATE.md updated to note this.

`scripts/phase0_score_seeds.py` SEED_DIM_MAP: `dependency_integrity → D5_wp_api` (was D8_errors). `build_seeds_judge_test.py` SEED_TO_JUDGE_FIELD: same correction.

## Step 0.1 — runbook (GPU container required)

`scripts/profile_base_model.py` now accepts `--adapter` (added 2026-05-11).

```bash
cd ~/Desktop/projects/wp-finetune
bash deps/dgx-toolbox/containers/ngc-pytorch.sh
# Inside container:
cd /workspace

# Base model — baseline routing without task-token bias
python -m scripts.profile_base_model \
  --model-path models/Qwen3-30B-A3B \
  --tokenizer-path adapters/tokenizer \
  --output-dir output/diagnostic/profiling_base \
  --subsample 0.05

# 30/70 adapter — measures whether task tokens actually drove routing concentration
python -m scripts.profile_base_model \
  --model-path models/Qwen3-30B-A3B \
  --tokenizer-path adapters/tokenizer \
  --adapter adapters/qwen3-30b-wp-30_70 \
  --output-dir output/diagnostic/profiling_30_70 \
  --subsample 0.05
```

Wall time estimate: ~20-30 min each on GB10 at subsample 0.05.

**Expected signal:** if task tokens are doing their job, the 30/70 run should show `eeff_wp_gen` and `eeff_wp_judge` diverge per layer. Mean E_eff total being 69 across all 5 ratios in `output/triage_decision.md` suggested they did not — that's the load-bearing assumption behind the entire v2 retrain plan.

## Step 0.3 — runbook (GPU container required)

`output/diagnostic/seeds_as_judge_test.jsonl` (145 records) built with human-derived GT.

```bash
# Inside container — serve 30/70 adapter via vLLM
# (clone recipes/nemotron-3-nano-4b-bf16-vllm.yaml to wp-30_70-vllm.yaml,
#  point base_model: models/Qwen3-30B-A3B, lora: adapters/qwen3-30b-wp-30_70)
sparkrun start recipes/qwen3-30b-wp-30_70-vllm.yaml

# Eval against seed-derived GT
python -m eval.eval_judge \
  --test-jsonl output/diagnostic/seeds_as_judge_test.jsonl \
  --output output/diagnostic/judge_30_70_seed_spearman.json
```

**Phase 1 hinges on this number.** If 30/70 against human-derived GT comes back around 0.65–0.75, the "judge is broken vs humans" premise weakens and the dataset rebuild scope shrinks. Run this *before* committing to Phase 1 rebuild.

**Caveat:** `eval_judge.py` `_GT_FIELD_TO_DIM` map currently lists D1/D2/D4/D6/D7 only. To pick up the synthesized seed fields (`sql_safety`, `wp_api_usage`, `error_handling`, `code_structure`), expand `_GT_FIELD_TO_DIM` to include them. Carried as Phase 0.9.

## Phase 0 follow-up tasks — current state

- 0.4 install composer + phpstan + WordPressVIPMinimum + phpcs-security-audit ✅
- 0.5 fix N/A heuristics (relaxed regexes + `<?php` auto-wrap) ✅
- 0.6 extend `profile_base_model.py` with `--adapter` ✅
- 0.7 build `seeds_as_judge_test.jsonl` synthesizer ✅
- 0.8 re-run step 0.2 with full tooling + fixes ✅ (results show original gate fails)
- 0.9 expand `eval_judge.py` `_GT_FIELD_TO_DIM` for seed-derived GT fields ⬜ — needed before step 0.3 can score all 9 dims
- 0.10 implement 18 LLM-assisted checks (rubric §F.5) ⬜ — likely required for credible D4/D5/D9 calibration
- 0.11 PASS-anchor extraction (clean WP core + top plugins) ⬜ — must happen before Phase 1 calibration

## Decision required from user

Phase 0 step 0.2 results contradict the plan's gate. Two structural facts must be acknowledged before Phase 1:

1. **The seed-only calibration approach is insufficient** for D4 / D5 / D9. PASS-anchors (Phase 0.11) and LLM checks (Phase 0.10) are upstream of Phase 1a.
2. **The 30/70 judge-eval-vs-human-GT (step 0.3) is the cheap experiment that decides scope.** If 30/70 actually ranks defects close to humans, the v2 rebuild is overkill; if it doesn't, the plan is justified.
