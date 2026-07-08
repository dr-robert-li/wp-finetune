# CONTINUE HERE — SHIP v1.3 ENSEMBLE, Phase 11 compression/packaging (updated 2026-07-08).

## ★ 2026-07-08 FINAL OUTCOME (read this first, supersedes everything below)
- **SHIP DECISION: two-model pair → v1.3 3-seed median ENSEMBLE judge (rho 0.842) + v1.2 gen (codegen 0.4616).**
  Single-seed s1 (0.827) is the leaner fallback if 3× judge serve is unacceptable.
- **Gap-closure investigation CLOSED — judge rho 0.827 is a LOCAL OPTIMUM.** Tested all 3 levers to reach
  ceiling 0.984; all negative:
  - B capacity (rank64 + train_attn, 3ep): **0.662 — OVERFIT.** Prior rank32/MoE-only was regularization,
    NOT a codegen handicap; two-model split does NOT unlock free capacity.
  - A loss-reshape (`--loss json_weighted`): alpha 0.5→0.773, alpha 3.0→0.780. **Uniform CE (v1.3) is the peak.**
  - C data-cleaning: gap distributed mid-band, not label outliers (drop-worst-15 only +0.015). Dominated.
  - Wall confirmed for SFT-on-relabeled-data on Qwen3-30B-A3B. Ceiling-mover = stronger base (qwen3.6/3.7).
  - Evidence: `output/relabel/{gap_closure_summary,leverA_loss_result,leverB_capacity_result,residual_audit,eval_seed_curve}.json`.
    Self-consistency probe also LOST (T=0.7 median@5 = 0.802 < greedy 0.827): `output/relabel/selfconsistency.json`.
  - New code: `scripts/reweight_json_loss.py` (self-check green), `--loss json_weighted` flag in `tinker_reasoning_sft.py`.
    Footgun fixed: `eval_relabel.py` wrote a fixed output path (clobbered v1.3 record) → now writes next to each capture.
- **NEXT: scaffold + plan Phase 11 compression/packaging.** No `.planning/phases/11-*` dir yet. Phase 11 must
  decide ensemble (3 LoRA seeds, heavier) vs single-seed s1 for the MoE-Sieve/AIMER target. Protected mask
  `output/.../protected_expert_mask.npy` (1,480 experts, Phase 7) carries reasoning through pruning — Sieve/AIMER must respect it.
- Kept-but-not-promoted Tinker ckpts from this session: `wp-reasoning-relabel-{capB-r64attn, leverA-jw3, leverA-jw0p5}`.

---

# (superseded) CONTINUE HERE — RL CLOSED (2026-07-05): ideal-conditions smoke killed on 6/6 G1 reads. v1.3 FINAL.

## ★ 2026-07-05 FINAL OUTCOME (read this first, supersedes everything below)
- **v1.3 = final judge artifact** (relabel 3-ep SFT seed1, rho 0.827 val / 0.841 holdout;
  `output/tinker/PROMOTED_v1.3.json`; merged at `models/_staging/qwen3-30b-wp-v1.3-merged`).
- **RL closed on the strongest possible evidence:** warm-start from v1.3 + oracle-PASSED calib-only
  reward (B2 CONDITIONAL-GO, defect stream zeroed) + 2 seeds + CI gates -> 6/6 checkpoint reads <=0,
  killed at steps 155/157 per pre-registered criterion. `output/rl_eval/SMOKE_V13_VERDICT.json`.
  Future RL requires a DIFFERENT signal family (execution-grounded / preference / multi-turn).
- **Two-model decision:** v1.3 judge + v1.2 gen (wpbench v1.3 0.381 < bar 0.4616; mix dose-response
  proved judge-rho tracks judge-exposure share — no mix recovers both). Phase 11 packages the pair.
- All QC belt-and-braces PASS: blinded disjoint reformat audit (Δ+2.1), holdout check (0.841),
  gate noise floors recorded (`gate_noise_floors.json`).
- Total Tinker spend across the whole campaign: ~$3-4 (measured $1.83 through v1.3 promotion).
- **NEXT: Phase 11 packaging (MoE-Sieve on the two-model pair per ROADMAP).**

# (superseded) seedA2 smoke DONE: KILLED on G1 (honest hybrid@0.8 failed the bar)

**Updated:** 2026-07-03 ~08:30 GMT+10 · **Branch:** `phase10-execution`

## ★ OUTCOME (read this first)
The hash-join-fixed rerun (`smoke_seedA2`) was the **first honest test of hybrid@0.8** —
calib provably in the loss (fired_frac 0.90–1.0 all run, step-0 gate green). Verdict: **KILL.**
- G1 step-50: ρ 0.6066 vs baseline 0.6243 (**−0.018**, bar +0.02), CI [−0.130,+0.082] ❌
- step-150 (informational trend): ρ 0.6434 (**+0.019**), CI [−0.059,+0.099] — under bar, in noise ❌
- **No Goodhart this time:** fc flat 0.28–0.34, entropy flat ~0.41, calib alive. The reward
  was honest; the signal is just too WEAK/SLOW to clear the gate. Real negative, not wiring.
- Full record: `logs/phase09_rerun/SMOKE_READS_TALLY.md` (seedA2 section + 07-01 re-label addendum).

**Recommendation, now doubly supported: hold RL, ship v1.2 SFT for v3.0.**
Any future RL needs a materially stronger per-step reward (grounded defect-detection term /
MO-GRPO separation — see 2026-07-02 session analysis), NOT more steps of hybrid@0.8.

## Process notes
- **Sentinel bug (recorded in tally):** step-50 watcher checked `checkpoint['step']` but manifest
  keys by `name` → run trained past the gate to step 161 (~110 unadjudicated steps) before the
  read. Banked step-100/150 ckpts converted part of that spend into the trend read.
- Run died silently at step 161 ~07:54 (no traceback; kill intent already issued on G1 fail).
- G2 codegen / G3 echo NOT read (moot after G1 kill). G2 required before any future continuation.
- All processes dead. Judge `wp-v4-judge-vllm` STOPPED (GPU free). No live jobs.

## What was fixed this session (commits on phase10-execution)
1. `fix(08.2 reward): GT hash-join was DEAD…` — canonical `normalized_code_hash` both sides,
   GT-coverage pool filter (482→342), calib telemetry (fired/mean/std/n per step), step-0
   CALIB_JOIN_DEAD halt, codegen trip-wire misconfiguration halt (+ `--codegen-probe-model-dir`),
   consistency weight-0 when unkeyed. Tests `tests/test_calib_join.py` 5/5.
2. `docs+launch(08.2)` — 07-01 tally re-label (pure-fc, hybrid untested), `launch_smoke_seedA2.sh`.
3. (this commit) seedA2 verdict + artifacts.

## Key artifacts
- Captures: `output/rl_eval/step-{50,150}-seedA2/judge_responses.jsonl` (n=121 each; 117/109 parseable)
- Metrics: `output/rl_checkpoints/smoke_seedA2/metrics/rl_metrics.jsonl` (161 steps, calib telemetry live)
- Summary: `output/rl_eval/rlev01_teacher_summary.json` (warmstart/50/150 aligned, n=86)
- 07-01 captures archived: `output/rl_eval_seedA1_0701/`
- Reusable: v4 save_state `tinker://d59dea4e-…/weights/wp-reasoning-v4-r32-rp30-savestate-final-state`

## 2026-07-03 SESSION 2 UPDATE — goal executed end-to-end
1. **SHIPPED:** v3.0 base = v1.2 SFT (ROADMAP amended, STATE current_phase=11, JOURNAL entry).
2. **Teacher ceiling MEASURED** (`08.2-TEACHER-CEILING.md`): stored-GT reliability 0.43 →
   ceiling 0.655; student 0.6243 = SATURATED; headroom +0.03 THIN. seedA2's bar was unclearable
   by construction. Branch resolved: **better teacher, not better reward.**
3. **Re-labeling protocol delivered** (`08.2-RELABEL-PROTOCOL.md`, from 107-agent deep research,
   22 verified claims): frozen anchored rubric, M=3 median-aggregated passes (Spearman–Brown on
   measured r=0.9025 → ceiling 0.983), active wave on disagreement items, κ/α + bias-audit QC
   gates, sentinel drift control, val re-labeled first. Next executable step = protocol step 0–1
   (pilot on val, in-session agents, $0).

## 2026-07-03 SESSION 3 UPDATE — re-label campaign EXECUTED (protocol steps 0-6)
- Rubric frozen (`scripts/relabel/RUBRIC_v1.md`); pilot QC ALL GATES PASS (rel 0.969, κ 0.623,
  verbosity 0.011, 0 drift flags); train wave 2×482 + active 3rd pass (34 items); **603/603
  items labeled**, `data/relabel_v1/labels.json` + `judge_gt_sidecar_v2.jsonl` (100% pool coverage).
- **Student TRUE gap: ρ 0.7477 vs new labels, ceiling ~0.98 → gap +0.24 ≥ 0.15 → ROUTE: SFT-FIRST.**
- Next training action: continued fine-tune of v1.2 on re-labeled judge data (rebuild the judge
  targets in openai_train from labels.json; new SFT stage; eval vs new val labels). RL only after.

## 2026-07-03 SESSION 4 — reformat audit + relabel-SFT proof (A/B isolated)
- **Reformat-probe bias audit: PASS** (belt-and-braces, 3rd QC axis after verbosity+sentinel).
  10 items (5 widest-dispersion + 5 random), whitespace-reformatted A/B, M=2 judges: Δ=0 all,
  no format bias. Scripts `scripts/relabel/reformat_probe_{prep,agg}.py`, `output/relabel/reformat_probe.json`.
- **Targets rebuilt** `scripts/relabel/rebuild_targets.py` → `data/reasoning_dataset/openai_train_relabel_v1.jsonl`:
  478 judge targets recalibrated from labels.json + median relabel dims (328 overalls changed,
  236 by ≥10pts); inline CoT dim numbers patched to match JSON; self-check green.
- **NOTE: trainer has NO load_state** (JOURNAL L127) — "continued FT of v1.2" is not a literal
  weight-continuation; it's a fresh 1-epoch LoRA-from-base on relabeled data. So ran the isolated A/B.
- **RESULT (same 118 val items, vs new labels, identical rank32/MoE-only/seed42/1-epoch recipe):**
  relabel-SFT (NEW) ρ=**0.638** [.508,.743] · old-ctrl (OLD) ρ=**0.487** [.318,.626] ·
  v1.2 mature ρ=0.757 [.660,.821]. **LABEL EFFECT = +0.152** (relabel−oldctrl) — relabeling helps,
  right direction. 1-epoch < mature v1.2 (fewer epochs). `output/relabel/eval_threeway.json`.
- **Next: promote via FULL (3-epoch) relabel SFT** (`--stage full --epochs 3` on the relabel jsonl)
  → expect to exceed 0.757 → new v1.3 judge. Ckpts: relabel sampler
  `tinker://717aeccb-…/sampler_weights/wp-reasoning-relabel-v1-ep1` (+ final-state savestate).
  Eval path: `capture_judge_responses_tinker.py` → `scripts/relabel/eval_relabel.py`.

## 2026-07-04 SESSION 5 — 3-epoch relabel SFT PROMOTED-READY (multi-seed confirmed)
- Gated on measured Tinker spend (human balance readings): Part1 3-epoch run + captures = **$0.74**
  (B0 102.59 -> B1 101.85). Real rate ~30x under prior parametric estimate; multi-seed ~$1.
- **Epoch curve (relabel SFT, n=121 vs new val labels):** ep1 0.610 · ep2 0.739 · ep3 **0.796** ·
  v1.2 bar 0.748. Monotonic, no collapse (loss 2.68, terse 0.0). `eval_epoch_curve.json`.
- **Multi-seed (3 seeds, ep3):** 0.796 / 0.827 / 0.790 → mean **0.804** sd 0.020; ALL 3 > v1.2.
  3-seed median-ensemble **0.842**. Paired adv CI [-0.007,+0.127] (grazes 0; robustness is the
  decisive evidence). `eval_multiseed.json`. Wired `--seed` into tinker_reasoning_sft.py.
- **DECISION: PROMOTE relabel 3-epoch SFT as v1.3 judge** (single seed s1=0.827, or ensemble 0.842
  if 3x serve cost OK). Ckpts: `tinker://ff1a9905-…/sampler_weights/wp-reasoning-relabel-v1-full-ep3`
  (+ final-state), seeds `…-relabel-s{1,2}` manifests. Re-measure student gap post-promote
  (ceiling 0.984, gap now ~0.14-0.18 → still SFT/thin; RL stays gated behind reward-v2).
- **✅ v1.3 PROMOTED (2026-07-04):** seed 1 ep3, rho 0.827. Canonical `output/tinker/PROMOTED_v1.3.json`,
  sampler `tinker://6a06e60f-…/sampler_weights/wp-reasoning-relabel-s1-ep3`, local export
  `models/tinker_export/wp-reasoning-v1.3`. STATE.md + JOURNAL + CHANGELOG updated. Total spend $1.83.
  NEXT: (opt) merge export for serving (Phase 11 packaging, GB10 memory wall); re-measure student gap
  on v1.3; adversarial reformat audit ($0) if desired.

## What's LEFT (user's call)
- **Accept:** route to v3.0 packaging on v1.2 SFT (Phase 11+ per ROADMAP).
- **OR** design reward-v2 (grounded defect-detection + MO-GRPO separation + anti-hack term),
  oracle-gate it offline (CI-lower>0), THEN one more gated smoke. The 07-02 analysis
  (conversation + tally addendum) has the full design sketch.
- Fix the two known stale scripts if rerunning: `launch_validated_smoke.sh` (both-seeds collision),
  `_rlev01_probe_ckpt.sh` (hardcoded RUN id).

## Watch-outs
- STATE.md frontmatter stale (trust this file + tally).
- Pre-existing test failures (env): test_lora_config, oracle-gate (stale captures), rl_judge_dispatch×3,
  preflight dotenv.
- `deps/dgx-toolbox` untracked = benign.
