# Phase 09 — Post-8.1 RL Rerun: PAUSE HANDOFF (resume with cleared context)

**Written:** 2026-06-25 ~07:50 AEST · **Branch:** `phase10-execution`
**Live launch plan (authoritative detail):** `09-LOCAL-RL-STATUS-UPDATES.md` §J
**Full prior context:** `09-LOCAL-RL-HANDOFF.md` (the FLAT 250-step run + $0 wiring)

---

## 0. TL;DR — where things stand

- **Phase 8.1 (reward redesign) is COMPLETE + verified + NYQUIST-COMPLIANT.** Offline gate PASS:
  judge `frac_mid` 0.011→0.691, `frac_groups_all_zero` 0.312→0.000, Phase-8 regression 62/62.
- Redesigned reward is **LIVE in the training path** (`rl_rollouts.py`: `_fix_score_from_completion`
  = 0.0 / 0.25 / rubric÷100 ; `judge_consistency_weight_lever2=0.45`, cap-0.5 guard intact).
- **RL rerun is LAUNCH-READY but NOT launched.** A free **dry-run validated the full training-loop
  wiring** (exit 0). Paused before paid Tinker compute because (a) context budget hit, (b) the
  step-1–3 wiring gate needs live supervision.
- **No active trainer.** Local judges UP ($0): wp_judge:8000 + wp_consistency:8001.

## 1. What was done this session

1. **Executed Phase 8.1 end-to-end** (4 plans, 3 waves): measure-first → diagnosis → reward fix
   (Lever 1 Form A + Lever 2) → offline gate. Verified 19/19 must-haves. Marked complete + validated.
   - Built `scripts/_gen_judge_probe_corpus.py` ($0 local-vLLM corpus generator); generated a fresh
     119-parseable / 93-G4-group corpus at `data/rl_probe/judge_probe_corpus.jsonl` (untracked).
2. **Closed the 04.2 dangling plan** (back-filled `04.2-01-SUMMARY.md`; plan-index now clean).
3. **`/gsd-validate-phase 08.1`** → NYQUIST-COMPLIANT (8/8 automated green).
4. **Prepped the Phase 9 rerun:** dry-run validated wiring; resolved `frac_mid` (derive at read-time
   = `1 − frac_reward_gt_0.9 − frac_reward_lt_0.1`); wrote launch plan to status-doc §J.

## 2. RESUME — exact next steps (supervised launch in fresh context)

Execute the launch plan in `09-LOCAL-RL-STATUS-UPDATES.md` §J. Summary:
1. Verify judges up (`curl -s localhost:8000/v1/models | grep wp_judge`; `:8001 | grep wp_consistency`).
   If down: `bash scripts/serve_v4_judge_vllm.sh` + `GPU_MEM_UTIL=0.22 bash scripts/serve_consistency_vllm.sh`.
2. **Arm OOM guard** (`scripts/_oom_guard.sh`, Monitor persistent) — DGX has no OOM protection. MANDATORY.
3. Start status-tick loop + telemetry monitor on the metrics JSONL.
4. Launch **two seeds** (42 / 7) — NOT two mechanisms (skill == launcher == same `rl_train.py`). Separate
   `--metrics-path` / `--manifest-path` / log / pid / `WP_JUDGE_DEBUG_DUMP` per seed (see §J for full cmd).
   `.venv-tinker/bin/python`; `set -a; . ./.env; set +a; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN`.
5. **EARLY WIRING GATE (step 1–3):** expect `frac_groups_all_zero → ~0.000` + varied per-sample fix
   scores (0/0.25/~1). If step-1 shows old ~0.31 / uniform groups → new reward NOT live → KILL + debug
   wiring (do not burn 50 steps).
6. **50-step kill/continue gate:** ≥1 seed must show `reward_min≠reward_max` + `reward_mean` trending up
   + frac_groups_all_zero low + frac_mid(derived)>0 + entropy stable. Continue winner → 500; kill loser.
   **Both flat → STOP both** and investigate the SEAM first (reward path live? warm-start baking old
   behavior? judge contention zeroing scores?) — NOT "Plan 03 wrong" (offline strongly validated it).

## 3. Cost

Only Tinker training compute (~8min/step; judges $0). Spend gate at 50 steps (checkpoint-every 50),
not at launch. ~7h/seed to the gate; full 500 ≈ 3 days.

## 4. Key pointers

- Launch detail: `09-LOCAL-RL-STATUS-UPDATES.md` §J · Prior FLAT run + $0 wiring: `09-LOCAL-RL-HANDOFF.md`
- Reward fix: `scripts/rl_rollouts.py` (`_fix_score_from_completion`, `judge_consistency_weight_lever2`)
- Measurement/gate artifacts: `.planning/phases/08.1-reward-redesign/08.1-{MEASUREMENT,SIGNAL-CHECK,VERIFICATION,VALIDATION}.md`
- Dry-run schema confirmed keys: frac_groups_all_zero/all_one/nonuniform, fix_correctness_mean/std,
  consistency_mean/std, group_reward_std_mean, entropy, frac_reward_gt_0.9/lt_0.1, reward_breakdown, reward_mean
- Untracked artifacts (git add if you want them tracked): `data/rl_probe/`, `logs/phase08.1/`, `logs/phase08.1/`

## 5. Open watch-items

- Offline-PASS is strong but the LIVE shape is unconfirmed — the step-1–3 gate is the real first test.
- DGX OOM unrecoverable; never >2 30B vLLM; keep OOM guard armed during any RL/eval.
- Each RL launch restarts from warm-start (NO resume) — a rerun is fresh, not a continuation.
- `.env` has ANTHROPIC_API_KEY — keep it out of any `claude -p` path (consistency is local vLLM now).
