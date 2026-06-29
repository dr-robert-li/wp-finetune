# RLEV-01/02 gate prep — step-500 (seedA) vs v1.2 SFT

Pre-staged during the monitor wait (2026-06-28/29). Run after step-500 lands.

## seedA identifiers
- Live run id: `tinker://9cb14129-f302-5c84-adf2-cc9ab92128a4:train:0/sampler_weights` (post-KL-fix).
  - OLD preKLfix run `03c69b7b` is DEAD — do not use.
- Checkpoints: step-50..500 every 50 (manifest.seedA.json).
- v1.2 SFT baseline model: `output/merge_v4_winner`.
- Warm-start init: `tinker://80c93d7c-...:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state`.

## RLEV-01 (no regression vs v1.2 SFT)
1. **Judge-Spearman trend** — `scripts/_rlev01_batch.py` (capture via Tinker, no vLLM) then
   `scripts/_rlev01_score.py` (teacher-GT Spearman). BOTH already extended CKPTS to step-500.
   - Verdict = 50->500 TREND + bootstrap CI lower > 0 (improved_beyond_noise), NOT point vs stale 0.1534.
   - ⚠ VERIFY warm-start path in `_rlev01_batch.py` (uses `...sampler_weights/...-ep3`) equals the
     init weights (`.../weights/...-final-state`) before trusting warmstart-relative bootstrap.
2. **wp-bench codegen HARD gate** — merged step-500 vs v1.2 SFT.
   - Bar: weighted overall (metadata.scores.overall) >= **0.4616** (NOT pre-SFT 0.4286).
   - Sub-floors: knowledge >= 0.45, execution >= 0.375.
   - v1.2 baseline result on disk:
     `output/eval_reasoning_v4_winner/revl04_rebench/reasoning_merged/wp_bench_results_20260613_214919.json`.
   - Needs: `scripts/merge_tinker_v3.py` (LoRA->merged) + vLLM serve + wp-bench run.
3. **9-dim**: no dimension regression vs v1.2 SFT.

## RLEV-02 (report + 5-part conjunctive gate) — `scripts/rlev02_report.py`
Five gates (ALL must pass): judge_spearman_improvement, wpbench_hard_gate,
antihack_no_reward_hack, protected_expert_retention, no_routing_collapse.
- **Anti-hack**: JSONLs already built — `output/antihack_validation/` (template_critique_collapse,
  self_preference_swap, verbose_padding). Run perturbed-RL + clean-v12 through judge endpoint;
  gate = hi_perturbed_rl < lo_clean_v12. Never hard-code fixture rewards.
- ⚠ **protected_expert_retention**: DO NOT feed rl_metrics `jaccard_protected` — it is **0.0 at every
  step by construction**. `rl_train.py:1042` hardcodes `active = np.zeros((48,128))` (Tinker remote
  exposes no per-step routing), so the in-loop jaccard is a STUB, not a real measurement.
  Real retention = POST-HOC router profiling of merged step-500 vs Phase-7 protected mask
  (`output/profiling/reasoning-merged-v4/protected_expert_mask.npy`). bar=0.85 provisional.
- reward-convergence curves + gen/judge delta: from rl_metrics.seedA.jsonl (real).
- router-shift log: per-step shift ratios (also needs real routing → post-hoc profiling).

## Training health at monitor close (reads #6-#9, all n=80, echo 0.25 PASS)
Judge within-run plateau: fixed-50 0.385 -> 250 0.413 -> 300 0.410 -> 350 0.413 -> 400 0.410 ->
450 0.403. Sustained ~+0.02-0.03 over fixed-50. KL pinned ~0.009, efrac ~0.96, halt=null, judges up.
Binding gate PASSED at 250 (seed12345 +0.027, seed99999 +0.055). Post-250 reads are CONFIRMING trend.

## EXECUTION STATUS (2026-06-30, live)
- Monitor + all 10 trend reads DONE (READS_TALLY). Endpoint judge fixed-50 0.385 -> 500 0.413 (+0.028 plateau). Training COMPLETE.
- Adapter EXPORTED: `models/tinker_export/seedA-step500/checkpoint.tar.gz/checkpoint.tar` (1.76G LoRA, sampler_weights/step-500 exports cleanly).
- RLEV-01 judge-Spearman: `_rlev01_batch.py` captures RUNNING (Tinker, offline canonical corr=None as expected); teacher-Spearman via `_rlev01_score.py` after captures finish.
- Base model present: `models/Qwen3-30B-A3B` (57G).

### CORRECTION — anti-hack gate (advisor-confirmed)
- My first run `build_antihack_set --score-and-gate` against :8000 = the **Phase-8 reward-infra
  robustness check** (single FROZEN canonical wp_judge), NOT the RLEV-02 RL-vs-v12 gate.
  :8000 wp_judge = "frozen wp_judge canonical checkpoint" (08-04-SUMMARY) = neither v1.2 nor step-500.
- That run's result (KEEP as reward-infra artifact, `acceptance_report.json`; fixture backed up to
  `acceptance_report.fixture.bak.json`): all 3 axes FAIL (perturbed > clean) on the frozen reward model
  — verbose_padding hi_p 0.363 vs lo_c -0.110; template_critique_collapse 0.428 vs -0.281;
  self_preference_swap 0.360 vs -0.160. NOTE: scalar = MO-GRPO z-normalized composite (relative, not
  absolute). This is a property of the reward MODEL, invariant to step-500; NOT the gate verdict.
  Consistency: echo-adversary held 0.25 PASS + reward modest across all 500 steps -> policy never
  exploited it. One absolute number can't say if RL *changed* hackability -> gate MUST be a comparison.
- CORRECT RLEV-02 anti-hack (10-RESEARCH ~543): score the 45-case set through TWO judge endpoints —
  (a) merged step-500 served + prompted as judge -> RL CI; (b) merged v1.2 served as judge -> v12 CI.
  Compare via `rlev02_report.check_antihack_gate(perturbed_rl, clean_v12)`. Question = "did RL make it
  MORE hackable than v12," not "is it hackable in absolute."

### REMAINING SERVE SEQUENCE (GB10 128GB unified; serve one 30B at a time)
0. Free frozen judges :8000/:8001 (not needed; rlev01_batch is Tinker-only) -> headroom for merge/serve.
1. Merge step-500: `merge_tinker_v3.py --adapter-tar <above> --base models/Qwen3-30B-A3B`.
2. Serve merged step-500 -> (a) wp-bench via `run_eval_reasoning.py --wpbench-only` (vs cached v1.2
   0.4616, sub-floors knowledge>=0.45/exec>=0.375); (b) anti-hack 45 cases as judge -> RL CI. Stop.
3. Merge + serve v1.2 (warmstart final-state, export from `.../weights/...-final-state`) -> anti-hack
   45 cases as judge -> v12 CI. Stop.
4. `rlev02_report.py` 5-part gate: wpbench + judge-spearman + antihack(RL-vs-v12). Retention +
   routing-collapse = NOT MEASURABLE this run (jaccard is zeros-stub) -> report as Wave-1 post-hoc TODO.

## Phase-10 workflow note
Full RLEV-01/02 = phase-10 Wave 1 (`/gsd-execute-phase 10` Tasks 4-7), gated behind the Task-3
HUMAN checkpoint ("live run landed"). v3.0 disposition decision stays with Dr. Li. This goal runs the
technical RLEV-01/02 comparison to produce the report for that human review.
