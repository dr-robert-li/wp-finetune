# Phase 09 — Local-Consistency RL Run: FULL CONTEXT HANDOFF

**Written:** 2026-06-24 ~22:25 AEST · **Branch:** `phase10-execution` · **Author:** Dr. Robert Li (w/ Claude)
**Purpose:** resume with a CLEARED context window. This is exhaustive and self-contained.
**Live status feed (watch remotely):** `.planning/phases/09-gspo-training/09-LOCAL-RL-STATUS-UPDATES.md`

---

## 0. TL;DR — where things stand RIGHT NOW

- **Local $0 consistency judge is WIRED, validated, committed (`abfe6b7`), pushed.** The paid `claude -p`
  consistency reward was replaced by a LOCAL vLLM model. RL is now fully $0 except Tinker training compute.
- **Full 500-step GSPO RL run is LIVE** (Tinker, warm-started MoE-only from v4). At ~step 231/500.
- **Reward is FLAT / not learning** (~0.27 plateau; decisive 151-200 window came in LOW at 0.247).
  Mechanically perfect (no halts/errors, guards healthy) but no optimization signal.
- **Root cause (evidence-based):** the dominant 0.7 `fix_correctness` reward term is BINARY (0 or 1, no
  middle) → GSPO groups go ~uniform → advantage collapses → vanishing gradient. NOT a bug, NOT under-exploration.
- **A step-250 STOP decision is ARMED** (scheduled wakeup): if still flat at step 250 → confirm checkpoint in
  manifest → clean-stop the trainer → run RLEV-01 fixed-set eval across checkpoints → decide signal-vs-redesign.
- **Future-run telemetry requirements written** to `09-RL-LOGGING-REQS.md`.

---

## 1. What this session built (the $0 consistency wiring)

**Problem:** the score-reasoning-consistency reward (0.3 weight, D-09-05) was scored via `scripts/claude_agent.py`
= `claude -p`, which per the 2026-06-22 policy ALWAYS bills paid API (~$90 was burned in an earlier signal run).
**Solution (09-HANDOFF Option 1):** repoint it to a LOCAL vLLM endpoint.

Files changed (committed in `abfe6b7`):
- `scripts/rl_judge_dispatch.py` — `score_judge_consistency_batch(... base_url=None)`. When `base_url` set,
  routes to local vLLM (`_score_via_vllm` + robust `_parse_consistency_score`: strips `<think>`, JSON→regex→bare
  float; enable_thinking=False, temp 0.2, max_tokens 256). Legacy claude path preserved at `base_url=None`.
- `scripts/rl_train.py` — added `--consistency-base-url` flag.
- `scripts/rl_rollouts.py` — passes `base_url=getattr(args,"consistency_base_url",None)` through `collect_rollouts`.
- `scripts/serve_consistency_vllm.sh` — serves the consistency model (see §3).
- `scripts/_oom_guard.sh` — DGX OOM watchdog (see §4).
- `scripts/_rl_status_tick.py` — one monitoring tick (metrics + containers + judge-quality spot-check → appends to status doc).
- Docs: JOURNAL.md entry, 09-HANDOFF.md "RESOLVED" addendum, 09-LOCAL-RL-STATUS-UPDATES.md (live feed).

**Model decision (IMPORTANT deviation):** the original 09-HANDOFF named `nvidia/Nemotron-3-Nano-Omni-30B-A3B-
Reasoning-NVFP4` — that is a VISION (Omni) multimodal model, wrong for a text 0-1 score and heavier on memory.
Used the TEXT-ONLY sibling **`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`** (NemotronHForCausalLM, 19.4GB, no
vision). User-confirmed. Both are vLLM-servable; text is strictly better here.

**Validation (all passed):** offline unit smoke; live signal run (warm-start gate green, step-0 reward
non-uniform, consistency endpoint hit +20 req/step, $0); throughput (warm 0.345s/call, batch-8 0.9s, decode
54.6 tps); no truncation at max-model-len 12288; judge-quality good 0.8-1.0 vs wrong 0.0.

---

## 2. The live RL run — config & exact relaunch command

**Process:** pid in `output/rl_checkpoints/rl_run.pid` (was 1540083). Launched as a harness background task.
**Warm-start:** MoE-only (D-09-08), v4 savestate `tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/
wp-reasoning-v4-r32-rp30-savestate-final-state` → log shows `train_mlp=True attn=False unembed=False`.
**Logs:** `output/rl_checkpoints/full_run.log`. **Metrics:** `output/rl_checkpoints/metrics/rl_metrics.jsonl`.
**Manifest:** `output/rl_checkpoints/checkpoint_manifest.json`. **Judge-fail dump:** `output/rl_checkpoints/judge_failures.jsonl`.
**Cadence:** ~8 min/step → 500 ≈ ~3 days total.

Exact launch (re-runnable; each launch RESTARTS from warm-start, NO resume):
```bash
cd /home/robert_li/Desktop/projects/wp-finetune
set -a; . ./.env; set +a; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN   # hygiene; Option 1 has no claude path
INIT="tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state"
env WP_JUDGE_DEBUG_DUMP=output/rl_checkpoints/judge_failures.jsonl \
  .venv-tinker/bin/python scripts/rl_train.py \
  --init-from "$INIT" --model-id Qwen/Qwen3-30B-A3B --lora-rank 32 --lora-seed 42 \
  --total-steps 500 --batch-size 8 --checkpoint-every 50 --jaccard-every 20 \
  --kl-soft 0.1 --kl-hard 0.3 --efrac-soft 0.7 --efrac-hard 0.5 \
  --judge-base-url http://localhost:8000/v1 --judge-model wp_judge \
  --consistency-base-url http://localhost:8001/v1 --consistency-model wp_consistency \
  --metrics-path output/rl_checkpoints/metrics/rl_metrics.jsonl \
  --manifest-path output/rl_checkpoints/checkpoint_manifest.json \
  > output/rl_checkpoints/full_run.log 2>&1
```
Then write the pid: `pgrep -f "[p]ython scripts/rl_train.py" > output/rl_checkpoints/rl_run.pid`.

**Sampling params:** temperature=1.0 (default, NOT lowered), group_size=4 (`rl_rollouts._build_sampling_params`).

---

## 3. Serving the judge + consistency models (both must be UP for the run)

Two vLLM containers on the DGX (GB10, 122GB UNIFIED memory):
- **wp_judge** (:8000, fix-scoring) — `bash scripts/serve_v4_judge_vllm.sh` (model
  `models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4`, gpu-util 0.55, max-model-len 8192). ~8min load.
- **wp_consistency** (:8001, consistency) — `GPU_MEM_UTIL=0.22 bash scripts/serve_consistency_vllm.sh`
  (model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, served name `wp_consistency`, max-model-len 12288,
  moe-backend flashinfer_cutlass; `--reasoning-parser` omitted by design). Model is downloaded in HF cache.
Ready checks: `curl -s localhost:8000/v1/models | grep wp_judge` ; `curl -s localhost:8001/v1/models | grep wp_consistency`.

**MEMORY (critical, DGX has NO OOM protection — OOM = unrecoverable hang):** judge 0.55 + consistency 0.22 =
0.77 fits. Download model uses Xet which STALLS — always `HF_HUB_DISABLE_XET=1` for hf downloads here.
**DGX cannot host a 3rd 30B vLLM** — to run eval you must STOP these two servers first (see §6).

---

## 4. Background processes running (check/kill on resume)

| What | How it runs | Check | Stop |
|---|---|---|---|
| RL trainer | harness bg task; pid in rl_run.pid | `kill -0 $(cat output/rl_checkpoints/rl_run.pid)` | `kill -TERM` (clean, preserves ckpt; NEVER -9 mid-write) |
| OOM guard | `scripts/_oom_guard.sh` (Monitor, persistent) | `pgrep -f _oom_guard.sh` | TaskStop / pkill |
| Doc-monitor loop | nohup loop running `_rl_status_tick.py` every 20min | `pgrep -f _rl_status_tick` | pkill |

**OOM guard:** trips when MemAvailable < 2048MB → kills `scripts/rl_train.py` + `docker stop` both containers +
logs a `🛑 OOM-GUARD TRIPPED` block to the status doc. (Verified mem healthy: AnonPages ~9.7-10GB stable, no leak.)

---

## 5. The flat-reward finding (the whole point of the diagnosis)

Reward window means (run at step ~231): `[0-50]=0.266  [51-100]=0.278  [101-150]=0.274  [151-200]=0.247
[201-231]=0.266`. NO upward trend over 200+ steps; the decisive 151-200 window dipped BELOW baseline.
Guards all healthy throughout: kl_v1=0 (genuine on-policy — see below), e_frac ~0.96 (halt<0.5, flat/safe),
0 errors, no halts, checkpoints 50/100/150/200 all wrote.

**kl_v1=0 is NOT a bug** (Section F of status doc): `kl_sample_train_v1` measures sample-vs-train logprob
staleness (an off-policy tripwire feeding the autohalt), ~0 by construction when on-policy. The actual drift
guard is `e_frac` (MoE routing health, HARD halt if <0.5; currently 0.96, flat). e_max_violation (~6.4) is a
MoE load-balance diagnostic, not a halt guard.

**Root cause — REWARD SHAPE (Section H, evidence-based):**
- fix_correctness (0.7 weight) is effectively BINARY: parsing all Panickssery divergent-rollout log lines
  (n=15) → frac<0.1=0.53, frac>0.9=0.47, **frac_mid=0.00**. A step function, no slope.
- consistency (0.3 weight) is graded (frac_mid=0.73) but too small to drive learning.
- Mechanism: 4 samples of a "fixable" prompt all score ~1, an "unfixable" prompt all ~0 → uniform GSPO groups
  → normalized advantage ~0 → vanishing gradient. Flat mean ≈ (frac_fixable·0.7 + consistency·0.3), set by
  prompt mix, not policy. Temperature=1.0/group_size=4 → exploration is fine (low-temp hypothesis REFUTED).
- NOT directly measurable this run: `frac_groups_all_zero` (per-group collapse) isn't logged — that gap is
  exactly what 09-RL-LOGGING-REQS.md fixes for the next run.

---

## 6. ARMED: step-250 stop + RLEV-01 protocol (Sections G/I of status doc)

A scheduled wakeup hops toward step 250. When step >= 250:
1. Compute window means incl `[201-250]`. "Learning" = window[201-250] clearly above ~0.275 plateau (rise
   >~0.02, trending up). Almost certainly FLAT given the trajectory.
2. **If flat/falling:** FIRST confirm step-250 checkpoint IS in `checkpoint_manifest.json` (auto-writes at 250;
   wait + recheck if absent). Do NOT stop until confirmed. Then `kill -TERM $(cat output/rl_checkpoints/rl_run.pid)`.
3. **RLEV-01 fixed-set eval** on warmstart + step-50/100/150/200/250 (whatever saved). PRIMARY discriminator =
   judge-Spearman on `data/reasoning_dataset/openai_val.jsonl` via `eval/eval_judge.run_eval` with
   `gt_mode=calibrated_canonical`, per checkpoint vs the warmstart/v1.2 baseline.
   - CONSTRAINT: no 3rd 30B vLLM fits → STOP wp_judge:8000 + wp_consistency:8001 first to free unified memory,
     serve the eval model at :8020. Checkpoints are Tinker LoRA sampler_weights → merge via `merge_tinker_v3.py`
     MoE-only, OR (lighter) sample each checkpoint via the Tinker sampling_client on the val set then offline-judge
     (avoids 6 merges). See Phase 10 `10-01-PLAN.md` for the full eval pipeline (eval_gen + eval_judge + wp-bench +
     `scripts/bootstrap_gate.py` `bootstrap_spearman_improvement`).
4. **DECISION:** ANY marginal Spearman improvement across checkpoints → recipe HAS signal → targeted RERUN with
   the 09-RL-LOGGING-REQS diagnostics added. FLAT Spearman → **REWARD REDESIGN before any further RL compute.**

---

## 7. Reward-redesign guidance (if RLEV-01 is flat — base on logged evidence)

The binary 0.7 fix_correctness term is the prime suspect. Candidate fixes (validate against the new per-group
/ component logging from 09-RL-LOGGING-REQS.md FIRST):
- **Graded partial credit** for fix_correctness instead of pass/fail (e.g. fraction of PHPCS/security/syntax
  sub-checks passed, weighted) → restores a smooth slope.
- **Rebalance weights** away from the saturated term (raise consistency's 0.3, or add the graded gen rewards).
- **Per-group diversity shaping / advantage floor** (the handoff's Mechanism-3 raw/non-normalized blend) so
  uniform groups still contribute gradient.
- Re-verify with `frac_groups_all_zero` once logged: if >0.5 at steady state, that's the confirmed kill signal.

---

## 8. Key file pointers
- Wiring: `scripts/rl_judge_dispatch.py`, `scripts/rl_train.py`, `scripts/rl_rollouts.py`
- Serve: `scripts/serve_consistency_vllm.sh` (:8001), `scripts/serve_v4_judge_vllm.sh` (:8000)
- Safety/monitoring: `scripts/_oom_guard.sh`, `scripts/_rl_status_tick.py`
- Live feed: `.planning/phases/09-gspo-training/09-LOCAL-RL-STATUS-UPDATES.md` (Sections A-I)
- Future logging spec: `.planning/phases/09-gspo-training/09-RL-LOGGING-REQS.md`
- Original phase handoff (Option-1 resolved addendum): `09-HANDOFF.md`
- Eval pipeline: `eval/eval_judge.py` (run_eval, _safe_spearman), `eval/eval_gen.py`, `scripts/bootstrap_gate.py`,
  `scripts/merge_tinker_v3.py`; Phase 10 plans `.planning/phases/10-rl-comparative-evaluation/10-01-PLAN.md`
- Probe/diagnostics: `scripts/_probe_rl_reward.py`
- Checkpoints (Tinker sampler_weights paths): `output/rl_checkpoints/checkpoint_manifest.json`
- Git HEAD: `abfe6b7 feat(09): $0 local-vLLM consistency judge (Option 1) + DGX OOM guard`

## 9. Resume checklist (cleared context)
1. Read this doc + the tail of `09-LOCAL-RL-STATUS-UPDATES.md` (latest D/F/G entries).
2. `kill -0 $(cat output/rl_checkpoints/rl_run.pid)` — is the trainer still alive? what step?
   (`tail output/rl_checkpoints/full_run.log`; window means via the python in §5.)
3. `docker ps` — are wp_judge:8000 + wp_consistency:8001 up? `pgrep -f _oom_guard.sh` — guard alive?
4. If run already STOPPED: check manifest for the last checkpoint, then proceed to §6 step 3 (RLEV-01 eval).
5. If still running + flat: the step-250 decision should fire automatically; otherwise execute §6 manually.
6. Decision tree: RLEV-01 improvement → targeted rerun w/ §1+09-RL-LOGGING-REQS diagnostics; flat → §7 redesign.

## 10. Open watch-items
- Reward flat (the headline). Mem healthy (no leak). kl_v1=0 benign. e_frac flat/safe.
- `.env` contains `ANTHROPIC_API_KEY` — Option 1/2 don't call claude, but don't source it into any `claude -p` path.
- Each RL launch restarts from warm-start (NO resume) — a rerun is a fresh run, not a continuation.
- DGX OOM is unrecoverable; never run >2 30B vLLM; always keep the OOM guard armed during any RL/eval run.
