# Continue Here — after Phase 08.2 (Reward Validity Gate)

**Paused:** 2026-07-01 (context exhaustion) · **Branch:** `phase10-execution` (pushed, in sync)

## TL;DR
Phase 08.2 (Reward Validity Gate) is **COMPLETE + VERIFIED (5/5) + committed + pushed**. It built the
reward-validity machinery and produced a decisive-but-humble result: **the old reward is proven dead;
no offline-safe replacement was found.** The phase does NOT green-light an RL rerun. Next decision is
the user's: run ONE gated smoke, or hold RL pending a non-code-blind reward.

## Where things stand
- **Phase 10 (RLEV):** DONE earlier this session. Verdict: seedA RL Goodharted → **reject RL, ship v1.2
  SFT for v3.0** (Dr. Li's disposition). Report: `logs/phase09_rerun/RLEV_FINAL_REPORT.md`.
- **Phase 08.2:** COMPLETE. STATE marks next_phase=09. 5 plans / 4 waves, all offline/CPU.
  - SC1 oracle standing gate: `scripts/reward_validity_gate.py` + `08.2-GATE-RULE.md` (+ test).
  - SC2 calibration reward (per-group pairwise-rank-agreement vs TRAIN GT): `scripts/reward_calibration.py`,
    wired in `scripts/rl_rollouts.py` (calib_weight=0 default = byte-identical). `rl_train.py` now has
    `--calib-form`/`--calib-weight`.
  - SC3 codegen trip-wire: `scripts/rl_codegen_tripwire.py` wired to `rl_train.py` halt seam;
    `--codegen-probe-every`/`--codegen-bar` (default 0 = off). Halts < v1.2 0.4616.
  - SC4 sweep: `scripts/reward_form_sweep.py` → `output/reward_validity/sweep_results.json`;
    `08.2-SWEEP-SELECTION.md`.
  - SC5 gated smoke SPEC (NOT executed): `08.2-SMOKE-RUNBOOK.md` + `scripts/launch_validated_smoke.sh`
    (dry-print/guarded).

## THE KEY FINDINGS (do not lose)
1. **Oracle is a REJECTER, not a VALIDATOR** (advisor-confirmed; baked into `08.2-GATE-RULE.md`).
   `fix_correctness` corr −0.24 → proven Goodhart (trust the FAIL). `pairwise_rank_agreement` "VALID"
   (+0.70) is BY CONSTRUCTION (monotone transform of same held-out data) → does NOT prove it works for
   training. Only the live smoke's within-run PAIRED teacher-Spearman trend can affirm.
2. **Sweep selected=null.** Only oracle-valid + gradient-alive configs = hybrid/pairwise @ calib_weight
   ~0.8 (ci_lo +0.37, frac_mid 0.74) — but they FAIL the worst-case echo proxy: pure calibration is
   CODE-BLIND. No offline-safe reward. The candidate for the smoke is `hybrid@0.8`, gated by the REAL
   codegen trip-wire + live echo-adversary.

## REMAINING / next decision (USER's call — do NOT auto-run)
- **Option A — hold RL.** Ship v1.2 SFT for v3.0 (Phase-10 verdict). Do not rerun RL until a
  non-code-blind reward exists (calibration + a codegen/anti-hack term or a preserved verifiable floor).
- **Option B — run ONE gated smoke** (costs GPU/Tinker): follow `08.2-SMOKE-RUNBOOK.md`. Launch
  `hybrid@0.8` with `--codegen-probe-every>0` armed; **kill at step 50** unless live paired
  teacher-Spearman is moving; then 250-step binding gate. NO 500-step runs on faith.
  Guarded launcher: `scripts/launch_validated_smoke.sh` (needs explicit confirm flag to actually launch).

## Watch-outs
- HARD: 08.2 is offline. Any RL launch is real spend — Option B only, with the trip-wire armed.
- STATE.md frontmatter has had stale `current_phase` fields all session — trust this file + STATE's
  roadmap-evolution, not the frontmatter narrative.
- `deps/dgx-toolbox` shows benign `.claude/` cache dirtiness (submodule-internal; correctly left).
- Judges up: :8000 wp_judge (=v1.2), :8001 wp_consistency. Merged models in `models/_staging/` (do NOT
  promote step-500 merge — forward anchor 8/9).

## Resume commands
```bash
cd /home/robert_li/Desktop/projects/wp-finetune
set -a; . ./.env; set +a; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
# re-run the reward-validity gate on any candidate reward form (offline, $0):
REWARD_SKIP_PHPCS_ASSERT=1 PYTHONPATH=. .venv-tinker/bin/python -c \
 "import scripts.reward_validity_gate as g; print(g.run_validity_gate('pairwise_rank_agreement'))"
# re-run the sweep:
REWARD_SKIP_PHPCS_ASSERT=1 PYTHONPATH=. .venv-tinker/bin/python -c \
 "import scripts.reward_form_sweep as s,json; print(json.dumps(s.select_config(s.run_sweep())))"
# gated smoke (Option B): read 08.2-SMOKE-RUNBOOK.md, then scripts/launch_validated_smoke.sh (dry-print first)
```

## Reference
- `.planning/phases/08.2-reward-validity/` — all plans/SUMMARYs/VERIFICATION/GATE-RULE/SWEEP-SELECTION/SMOKE-RUNBOOK.
- `output/reward_validity/{ORACLE_FINDING.md,sweep_results.json}` — the numbers.
- `logs/phase09_rerun/RLEV_FINAL_REPORT.md` — Phase-10 reject-RL verdict.
- JOURNAL.md top two entries (2026-06-30 Goodhart, 2026-07-01 probe) — the narrative.
