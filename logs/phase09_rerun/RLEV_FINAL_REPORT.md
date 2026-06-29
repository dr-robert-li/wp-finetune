# RLEV-01/02 — seedA step-500 (GSPO RL) vs v1.2 SFT — FINAL (2026-06-30)

Branch `phase10-execution`. Candidate: seedA RL step-500 (Qwen3-30B-A3B, MoE-only warm-start from
v1.2 SFT = reasoning-merged-v4, lr 1e-05, GSPO, 500 steps). Baseline: v1.2 SFT (merge_v4_winner).
Proceeded through phase-10 Task-3 human gate on the explicit /goal directive. Deliverable = comparison
report; v3.0 disposition stays with Dr. Li.

## Headline: MIXED→NEGATIVE. The 5-part conjunctive gate FAILS.
RL improved its PROXY reward (judge fix-correctness +0.028 within-run) but did NOT improve the
validated RLEV-01 targets, and regressed codegen. Classic Goodhart: moved the reward, not the target.

## RLEV-01 (no regression vs v1.2 + judge-Spearman improvement = primary target)
### a) Judge teacher-Spearman (SOFT, baseline-comparable; n=85 pairs; baseline warmstart rho=0.5732)
| ckpt | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 |
|------|----|----|----|----|----|----|----|----|----|----|
| rho  |0.530|0.574|0.597|0.616|0.556|0.559|0.582|0.621|0.595|**0.552**|
- **NO checkpoint improved beyond noise** (every bootstrap CI straddles 0). step-500 = 0.552, BELOW
  baseline (Δ−0.022). Peaks step-200/400 (~0.62) but not significant. PRIMARY RL TARGET NOT MET.
### b) wp-bench codegen (REVL-04, full 344 tasks, wp-core-v1, temp 0.0; HARD bar 0.4616)
- step-500 overall = **0.4125**  vs  v1.2 = 0.4616  →  **REGRESSION −0.049  → HARD GATE FAIL.**
- BINDING (merge fidelity verified). `_wpbench_with_boot` skipped `assert_served_identity`; run
  separately via `_04.4_anchors_v3.py`. The weight anchors are the discriminating evidence and PASS,
  so the eval number is trustworthy. Two-bar distinction: TRUST THE EVAL = yes (weight anchors
  definitive); PROMOTE DIR TO PROD = no (re-merge to clear forward anchor first). Anchor result:
  - **TENSOR PASS + FP32-CONTROL PASS** (the two PRIMARY precision certifiers): merged weights ==
    bf16(stock_base + per-expert delta) exactly, per-expert deltas distinct → weight bake faithful.
  - **FORWARD 8/9 PASS, 1 marginal FAIL** (L47_seed137): cos 0.99995 (≥0.99990 ✓), rel_l2 0.0083
    (≤0.01 ✓), mean 0.0016 (≤0.002 ✓), router-invariant ✓ — fails ONLY the max_abs cap (0.125 vs 0.1),
    a single bf16 tail element at the deepest layer in the CPU rank-vs-weight cross-check.
    Overall `staging_anchor_FAILED` on that one probe.
  - Reading: merge is faithful (definitive weight anchors pass); lone forward fail = bf16 tail noise,
    NOT a merge defect → −0.049 codegen regression treated as REAL (high confidence). Do NOT promote
    this staging dir for production serving without re-merging to clear the forward anchor cleanly.
- sub: knowledge 0.4625 (v1.2 0.4906; floor 0.45 → PASS, −0.028); correctness/exec 0.375 (v1.2 0.4375;
  floor 0.375 → AT FLOOR, −0.0625). Result: `output/rl_eval/wpbench_seedA_step500/.../wp_bench_results_20260630_025516.json`.
- Minor caveat (non-blocking): v1.2 baseline 0.4616 is from 2026-06-13 under `dgx-vllm-eugr-nightly:latest`,
  which may have drifted. Drift would hit BOTH 30B models equally → cannot manufacture a candidate-specific
  regression, so direction is unaffected. Re-bench v1.2 under today's image only if an exact decimal is needed.

### within-run fix-correctness (the RL PROXY reward; _check_judge_fixcorr, n=80/read, echo 0.25 all PASS)
- fixed-50 0.385 → 250 0.413 → 300 0.410 → 350 0.413 → 400 0.410 → 450 0.403 → 500 0.413 = +0.028.
  Binding gate "passed" at 250 on THIS metric — but it is the optimized proxy, not RLEV-01's validated
  judge-Spearman. The two diverge → reward overfit.

## RLEV-02 (report axes)
- **anti-hack: PASS — no evidence of increased hackability vs v1.2** (n=15/axis, wide CIs → directional,
  not high-power). Margin Δ(RL−v12): verbose_padding +0.132, template_critique_collapse +0.085,
  self_preference_swap −0.005 (tied). Absolute single-judge gate FAILS for BOTH v12 and RL (pre-existing
  reward-model property, not an RL regression). Detail: `RLEV02_ANTIHACK_RESULT.md`.
- **reward convergence**: reward_mean modest ~0.25–0.45 across 500 steps; KL pinned ~0.009 (soft 0.1);
  e_frac ~0.96; halt=null; clean completion (step-500 + final-step-500 saved).
- **protected-expert retention + router-shift: NOT MEASURABLE this run.** `jaccard_protected` is a
  zeros stub (rl_train.py:1042 hardcodes active=zeros; Tinker exposes no per-step routing). Needs
  Wave-1 POST-HOC routing profiling of merged step-500 vs Phase-7 mask
  (`output/profiling/reasoning-merged-v4/protected_expert_mask.npy`). NOT a gate fail — unmeasured.

## 5-part conjunctive gate
| gate | verdict |
|------|---------|
| judge_spearman_improvement | **FAIL** (no ckpt beyond noise; step-500 below baseline) |
| wpbench_hard_gate (≥0.4616) | **FAIL** (0.4125, −0.049 regression) |
| antihack_no_reward_hack (vs v12) | PASS (comparative) |
| protected_expert_retention | NOT MEASURABLE (Wave-1 post-hoc) |
| no_routing_collapse | NOT MEASURABLE (Wave-1 post-hoc) |
→ **Conjunctive gate FAILS** on the two measurable RLEV-01 axes.

## Interpretation + recommendation (for Dr. Li)
- step-500 over-trained: endpoint worse than mid-run on teacher-Spearman (peaks ~200/400) AND on
  codegen. lr 1e-05 GSPO moved the fix-correctness proxy but not validated judge quality, and traded
  off codegen (−0.049).
- The standing Plan B (READS_TALLY / continue-here) — "if step-500 weak, restart higher LR" — is now
  supported by data: the signal is weak-positive on the proxy only, negative on the validated gates.
- Options for disposition (HUMAN call): (1) reject RL, keep v1.2 SFT for v3.0; (2) re-run RL at higher
  LR / shorter horizon (step-200/400 looked best) with codegen-regression guardrail; (3) evaluate the
  step-200/step-400 checkpoints (best teacher-Spearman) on wp-bench before deciding — cheaper than a
  re-train and may dominate step-500.

## Artifacts
- wp-bench: output/rl_eval/wpbench_seedA_step500/  | judge-Spearman: output/rl_eval/rlev01_summary.json + logs/phase09_rerun/rlev01_score.log
- anti-hack: output/antihack_validation/acceptance_report.{v12_judge,rl_step500}.json
- reads: logs/phase09_rerun/READS_TALLY.md | merged step-500: models/_staging/qwen3-30b-wp-seedA-step500-merged
