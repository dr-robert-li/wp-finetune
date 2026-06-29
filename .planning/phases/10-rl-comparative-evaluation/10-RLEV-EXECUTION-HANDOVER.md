# Phase 10 — RLEV-01/02 Execution Handover (seedA step-500 vs v1.2 SFT)

**Date:** 2026-06-30 · **Branch:** `phase10-execution` · **Author:** Dr. Robert Li (Claude Code session)
**Status:** RLEV-01/02 gate RUN and REPORTED. Conjunctive gate **FAILS**. v1.2 SFT stands for v3.0
pending Dr. Li's disposition. Proceeded through the phase-10 Task-3 human gate on the strength of the
explicit session goal; the v3.0 promotion decision remains with Dr. Li.

This document is the single exhaustive record of the RLEV execution. Primary result files:
- `logs/phase09_rerun/RLEV_FINAL_REPORT.md` — the verdict + all numbers (authoritative).
- `logs/phase09_rerun/RLEV02_ANTIHACK_RESULT.md` — anti-hack RL-vs-v12 detail.
- `logs/phase09_rerun/RLEV_GATE_PREP.md` — pre-stage discovery + corrections.
- `logs/phase09_rerun/READS_TALLY.md` — all 10 controlled-eval reads (ground truth).

---

## 1. What was done
1. **Monitored seedA RL to step-500.** Detached Tinker run (PID 207480), warm-started MoE-only from
   v1.2 SFT (reasoning-merged-v4), lr 1e-05, GSPO, KL 0.1/0.3, efrac 0.7/0.5, judge_max_new_tokens=4096.
   Reached step-500 with a clean process exit; `step-500` + `final-step-500` checkpoints saved
   (`output/rl_checkpoints/metrics/manifest.seedA.json`). KL pinned ~0.009, e_frac ~0.96, halt=null,
   judge_failures stable (5→6) across the whole run.
2. **5 trend reads at 300/350/400/450/500** (`_check_judge_fixcorr`, n=80/read, temp 0.2, seed 12345),
   appended to READS_TALLY. Judge fix-correctness plateau: fixed-50 0.385 → 500 0.413 (+0.028);
   echo-adversary held 0.25 (PASS) every read. Binding gate had already PASSED at 250 in a prior session.
3. **Ran the RLEV-01/02 gate** on merged step-500 vs v1.2 SFT (see §2).

## 2. Results — conjunctive gate FAILS
| gate axis | verdict | numbers |
|---|---|---|
| RLEV-01 judge-Spearman (primary RL target) | **FAIL** | teacher-Spearman: no checkpoint improved beyond noise (n=85, all bootstrap CIs straddle 0); step-500 rho 0.552 < warmstart 0.573. Peaks step-200/400 (~0.62) but non-significant. Merge-INDEPENDENT (Tinker-sampled). |
| RLEV-01 wp-bench codegen (HARD ≥0.4616) | **FAIL** | step-500 overall 0.4125 vs v1.2 0.4616 = **−0.049 regression**. Sub: knowledge 0.4625 (floor 0.45 ✓), correctness 0.375 (=floor). Full 344 tasks, wp-core-v1, temp 0.0. BINDING (merge weight-anchors verified faithful). |
| RLEV-02 anti-hack (RL vs v1.2) | **PASS** | no evidence RL more hackable. Margin Δ(RL−v12): verbose_padding +0.132, template_critique_collapse +0.085, self_preference_swap −0.005. n=15/axis (directional). |
| RLEV-02 protected-expert retention | NOT MEASURABLE | `jaccard_protected` is a zeros stub (`rl_train.py:1042` hardcodes `active=np.zeros`; Tinker exposes no per-step routing). Needs Wave-1 post-hoc routing profiling vs Phase-7 mask. |
| RLEV-02 no-routing-collapse | NOT MEASURABLE | same stub; Wave-1 post-hoc. |

**Interpretation:** RL lifted its PROXY reward (judge fix-correctness +0.028) but not the VALIDATED
judge-Spearman, and regressed codegen −0.049 → classic Goodhart / over-training. The step-500 endpoint
is worse than mid-run (step-200/400) on both teacher-Spearman and (by inference) likely codegen.

## 3. Disposition options for Dr. Li (HUMAN decision)
1. **Reject RL, keep v1.2 SFT for v3.0** (clean; the gate failed on two measurable axes).
2. **Re-run RL at higher LR / shorter horizon** with a codegen-regression guardrail (Plan B from
   READS_TALLY — lr 1e-05 was conservative, KL pinned ~0.009 ≈ +0.00014/step; signal was weak-positive
   on the proxy only).
3. **wp-bench step-200/step-400 before deciding** — cheaper than a re-train; those checkpoints had the
   best teacher-Spearman and highest reward_mean and may dominate step-500. (NOT done here — parked as
   a recommendation; outside the "gate step-500" scope.)

## 4. How to reproduce / re-run the gate
Env: `cd <repo>; set -a; . ./.env; set +a; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN; export PYTHONPATH=.`
Venv: `.venv-tinker/bin/python` (peft + accelerate were installed this session for the merge).
- **Judge-Spearman:** `_rlev01_batch.py` (Tinker capture, no serve) → `_rlev01_score.py` (teacher-GT
  Spearman; bypasses the calibrated_canonical gate that zeroes out in this env). Both extended to step-500.
- **wp-bench:** `_rlev01_wpbench_step500.py` (wraps `run_eval_reasoning._wpbench_with_boot`; boots vLLM
  on the merged dir, runs REVL-04, stops). Compares to cached v1.2 0.4616.
- **anti-hack:** serve a model as `wp_judge` on :8000, then
  `python -m scripts.build_antihack_set --score-and-gate --judge-base-url http://localhost:8000/v1`.
  Run once per judge (v1.2 and step-500), compare margins. (NOTE: scores both perturbed+clean through
  ONE judge with MO-GRPO normalization → compare MARGINS across runs, not raw cross-run scalars. The
  literal `rlev02_report.check_antihack_gate` cross-mixes two normalizations — margin-delta used instead.)

## 5. Merge / serve mechanics (GB10, 128GB unified, one 30B at a time)
- **Adapter export:** `tinker_export_checkpoint.py --tinker-path <sampler_weights/step-500>` → 1.76GB
  LoRA tar at `models/tinker_export/seedA-step500/checkpoint.tar.gz/checkpoint.tar`. (seedA saved
  ONLY `sampler_weights/`, no `weights/` — export works on sampler_weights fine.)
- **Merge:** `merge_tinker_v3.py` (CPU, loads stock base bf16) →
  `models/_staging/qwen3-30b-wp-seedA-step500-merged` (13 shards). MoE deltas non-trivial
  (w1 0.0148 / w3 0.0139 / w2 0.0100). Required `peft`+`accelerate` (installed into .venv-tinker via uv).
- **Anchor certification** (`_04.4_anchors_v3.py`): TENSOR + FP32-CONTROL **PASS** (weight bake exact);
  FORWARD 8/9 PASS, 1 marginal FAIL (L47_seed137: cos 0.99995, rel_l2 0.0083, mean 0.0016 all pass —
  fails only max_abs 0.125 vs 0.1 cap = bf16 tail at deepest layer). Status `staging_anchor_FAILED` on
  that one probe. Reading: merge faithful (definitive weight anchors), eval number trustworthy.
  **DO NOT promote this staging dir to production serving without re-merging to clear the forward anchor.**
- **Serving:** `serve_v4_judge_vllm.sh` (override MODEL_DIR/CONTAINER_NAME/SERVED_NAME/PORT) for the
  judge endpoint; `_wpbench_with_boot` self-manages the wp-bench serve. `wp_judge` :8000 == v1.2 SFT
  (`merge_v4_winner`); `wp_consistency` :8001 == Nemotron-3-Nano-30B-NVFP4.

## 6. Infra state at handover (box left as found)
- Docker judges RESTORED + healthy: `wp-v4-judge-vllm` (:8000, v1.2 = wp_judge) and
  `wp-consistency-vllm` (:8001, wp_consistency). Both `AutoRemove=true`; restored via their serve scripts.
  Config snapshot saved at `logs/phase09_rerun/judge_restore/judge_containers_inspect.json`.
- During the gate both were freed to fit the step-500 serve; consistency OOM'd twice on restart under
  RAM pressure (page cache from merges) then loaded cleanly once cache was reclaimed.
- `peft==0.19.1` + `accelerate==1.14.0` + `psutil` added to `.venv-tinker` (merge dependency).

## 7. Caveats / watch-outs
- **jaccard_protected is a stub (0.0 always)** — never feed `rl_metrics` jaccard to a retention gate.
- **wp-bench baseline drift (non-blocking):** v1.2 0.4616 is from 2026-06-13 under
  `dgx-vllm-eugr-nightly:latest`; nightly may have drifted but hits both 30B models equally → cannot
  manufacture a candidate-specific regression. Re-bench v1.2 under today's image only for exact decimals.
- **Stale run id trap (fixed):** `_rlev01_batch.py` had hardcoded the OLD preKLfix run `03c69b7b`;
  repointed to live seedA `9cb14129`. Old `full_run.log` / `rl_metrics.seedA.jsonl.preKLfix` are the dead run.
- **Anti-hack absolute gate:** "hi_perturbed < lo_clean" FAILS for BOTH v12 and RL — a pre-existing
  reward-model-family property, NOT an RL regression. Only the RL-vs-v12 comparison is the gate.

## 8. New/changed files this session (committed)
- New: `scripts/_seedA_watch.sh`, `scripts/_rlev01_wpbench_step500.py`,
  `logs/phase09_rerun/{RLEV_FINAL_REPORT,RLEV02_ANTIHACK_RESULT,RLEV_GATE_PREP}.md`,
  this handover, `output/antihack_validation/acceptance_report.{v12_judge,rl_step500}.json` (under output/ —
  may be gitignored; numbers are in the reports regardless).
- Changed: `scripts/_rlev01_batch.py` (run id + CKPTS→500), `scripts/_rlev01_score.py` (CKPTS→500),
  `READS_TALLY.md` (reads #6–#10), phase-10 `.continue-here.md`, `output/rl_checkpoints/.../manifest.seedA.json`.
- NOT committed (large generated data, gitignored): `data/phase1_extraction/` (5.5G),
  `data/phase3_cot/` (68M), `graphify-out/` (7.5M), merged models + adapter tars under `models/`.
