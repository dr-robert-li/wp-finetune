Lock decisions A3/A4 as go (GB10 time is free). Lock decision for budget ceiling for D as go, use as much as required (Tinker is trivial). Review B2 oracle numbers with spawned background agent using Fable 5 model to determine go/no-go into Phase C

Phase A — Consolidate v1.3 (before any RL)

A1. Verify export landed (in flight, waiter live). Gate: tar valid + non-trivial. $0.

A2. Re-measure student gap on v1.3 — capture already exists (eval_s1_ep3); it's ρ=0.827 → gap-to-ceiling +0.157. Done, recorded in PROMOTED_v1.3.json. $0.

A3. Merge + serve v1.3 (Phase 11 packaging path): export → merge_tinker_v3.py → local merged model → vLLM. Needed for reward-time judge serving AND wp-bench. Watch: GB10 memory wall (documented), byte-identity check like v4-winner. $0 Tinker, ~hours GPU.

A4. wp-bench codegen check on merged v1.3. Rationale to run it now even though relabel is judge-only: this sets the new codegen bar for RL gates (v1.2 bar 0.4616 no longer the right reference if v1.3 ships). Gate: score ≥ 0.4616 − noise. If it regresses (unlikely, gen data unchanged) → investigate before promoting to serving.

A5 (optional robustness). Adversarial reformat audit — separate blinded judge agents, orig/reformat in disjoint batches, n≈30. $0 (in-session agents). Closes the "paired judges saw both variants" hole.

A6 (optional robustness). Hold-out label check: M=3 re-label of ~30 fresh items never seen by rubric agents → confirm v1.3's 0.827 isn't rubric-instrument overfit. $0 (agents) + 1 capture (~$0.15).

Phase B — Reward-v2 design + offline oracle gate (RL prerequisite; the 07-02 sketch)

B1. Build reward-v2 components:
- Grounded defect-detection term: reward = agreement with sidecar-v2 GT on specific defects (per-dim), not just overall-score proximity. Sidecar v2 already merged (100% pool coverage, judge_gt_sidecar_v2.jsonl; note loader asserts source=="train" — pass sidecar_version explicitly).
- MO-GRPO separation: calibration / format / codegen-guard as separate advantage streams, not one blended scalar (prevents the 70/30 composite Goodhart).
- Anti-hack term: perturbation-margin check (clean vs perturbed code must rank correctly) folded into reward or as trip-wire.
- Fix known stale wiring while in there: launch_validated_smoke.sh both-seeds collision, _rlev01_probe_ckpt.sh hardcoded RUN id, arm --codegen-probe-model-dir (the never-armed trip-wire).

B2. Offline oracle gate (hard, before ANY training): replay reward-v2 over existing captures (warmstart/step-50/step-150 + v1.3's) — reward must rank checkpoints in the same order as measured judge-ρ. Gate: Spearman(reward, judge-ρ) CI-lower > 0 across ≥3 checkpoints. $0 (offline). The 08.1 lesson: never train on a reward that hasn't demonstrated it tracks the target.

B3 (optional robustness). Noise-floor calibration of every gate: bootstrap each gate metric on existing data, set thresholds at ≥2 SE — the "point-bar that doesn't know its noise floor" lesson, twice-learned. $0.

Phase C — Gated RL smoke (only if B2 passes)

C1. Warm-start from v1.3 — create_training_client_from_state (exists in  s1: re-run seed-1 3-epoch with --save-state (~$0.25) OR use the
default-seed run's existing final-state (0.796, has save_state) as the whtly weaker init. Never fresh-LoRA into RL again (JOURNAL L127).

C2. Step-0 preflight gates (halt-on-fail, all already exist): CALIB_JOIN codegen trip-wire armed + reads, parse-fail averaged over 10 steps not
1 (SE ~14pts on single-step), sentinel watcher keyed by name not step (t

C3. Smoke ~150–200 steps, seeds A+B (collision fix from B1). Reads at 50
- G1 calibration: judge-ρ vs relabel GT, bar = warmstart + 0.02 with CI  at 0.827 vs ceiling 0.984 → headroom +0.157, ~5× the seedA2 headroom(+0.03) — the bar is now clearable by construction, which was the seedA2 kill's root cause.
- G2 codegen: ≥ A4 bar (hard, no-regression).
- G3 format: terse rate 0, echo/entropy flat.
- Kill criteria pre-registered: any Goodhart signature (fc↑ while ρ flatc collapse.

C4 (optional robustness). Third seed on the smoke winner config before s

Phase D — RL scale run (only if C passes on both seeds)

D1. 500-step run, banked ckpts every 50, watcher + trip-wire live, MO-GR Est. single-digit $.
D2. Checkpoint sweep eval (judge-ρ + wp-bench per banked ckpt) — pick pet endpoint ≠ best).
D3. Promotion gate = conjunctive: ρ beats v1.3's 0.827 (CI clear), codeg v1.3's. Multi-seed confirm on the winner (~$1).
D4. Promote as v1.4/v2.0-RL; same doc set (PROMOTED json, STATE, JOURNAL