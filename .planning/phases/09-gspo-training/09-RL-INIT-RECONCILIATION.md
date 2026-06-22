# Phase 09 — RL Init & Train-Flag Reconciliation (DECISION DOC, SIGNED OFF)

**Status:** 🟢 SIGNED OFF — Option A (RL MoE-only, supersedes D-09-02) approved by Dr. Robert Li 2026-06-22T16:47:00+10:00. D-09-08 recorded (ROADMAP/PROJECT/STATE). Execution plan (v4 `save_state` regen → warm-start relaunch) is now unblocked and awaits a go signal — no compute burning yet.
**Date:** 2026-06-22
**Author:** Claude (live-run monitoring loop)
**Trigger:** Live Phase 9 GSPO run found mis-initialized; root-caused to two upstream design issues.

---

## TL;DR — the decision you need to make

The live RL run was **cold-starting a fresh LoRA on the raw `Qwen/Qwen3-30B-A3B` base** — no v1.2 SFT weights — so it had zero WordPress/judge capability and would fail RLEV-01 by construction. The fix is to **warm-start RL from the v1.2 SFT LoRA**. Confirming feasibility surfaced two blockers:

1. **The v4-winner (the RLEV-01 baseline) has no loadable Tinker training checkpoint** — only `sampler_weights` (Tinker's `load_state` rejects those). A `save_state` `/weights/` checkpoint must be regenerated (re-run the r32-rp30 SFT with `--save-state`). v3 *does* have a loadable checkpoint but is the pre-v4 model (weaker codegen).
2. **Train-flag contradiction (the real decision):** the v4-winner SFT is **MoE-only** (`train_attn=False, train_unembed=False`), but the RL design **D-09-02 mandates `train_mlp=True, train_attn=True, train_unembed=True`**. These cannot both hold. **Pick the RL train-flag policy** (Option A/B/C below) — that choice also determines whether the warm-start is clean.

**Recommendation: Option A — RL trains MoE-only, formally superseding D-09-02's `train_attn/unembed=True`.** Rationale below.

---

## Background — how this was found

Decision chosen by user (2026-06-22): regenerate v4 `save_state`, then warm-start. This doc verifies reproducibility and surfaces the train-flag conflict that blocks a clean execution.

The live run (`rl_train.py --model-id Qwen/Qwen3-30B-A3B --lora-rank 32 --lora-seed 42`, sessions `f712a2b8`→`6114b687`→`4051044`) was monitored over ~36–41 steps. Symptoms:

- `reward_mean` flat/noisy (~0.25, no trend over 41 steps); `kl_sample_train_v1 = 0.0000` every step.
- Investigation ruled out code bugs (GSPO gradient path correct; KL=0 is an on-policy artifact; advantage variance present).
- Rollout failure dumps were dominated by **raw-instruct refusals** — *"I can't assist with that request"*, *"Could you clarify…"* — phrasing SFT trains out; impossible from a v1.2 policy at any rate.
- **Code confirmed the cause:** `build_training_client` → `create_lora_training_client(base_model=Qwen3-30B-A3B, rank=32, seed=42)` with **no `load_state`** anywhere. Fresh LoRA on raw base.

Run was killed at step ~42 (KL/reward never moved → zero real loss).

---

## Finding 1 — warm-start is required (not optional)

RLEV-01 (`REQUIREMENTS.md:172`): *"RL model evaluated against v1.2 SFT baseline … no dimension regression permitted; judge Spearman improvement expected."* Phase 10 (`10-CONTEXT.md:9`) compares the RL model against the **v4-winner baseline** (`output/merge_v4_winner`). A model RL'd from raw base has no WordPress/judge capability → catastrophic regression vs v1.2 → RLEV-01 fails by construction. **RL must start from the v1.2 SFT weights.**

## Finding 2 — warm-start is feasible, but the ideal source isn't loadable

- Tinker supports warm-start: `ServiceClient.create_training_client_from_state(path)` derives `base_model`/`rank`/`train_*` from the checkpoint and `load_state()`s it (router stays frozen — no `train_router` arg). ✓ Mechanism wired into `rl_train.py` as `--init-from` (uncommitted; diff reviewed).
- **v4-winner (`fc55e8b9 … wp-reasoning-v4-r32-rp30`) — NOT loadable.** Manifest `state_path: null`; only `sampler_weights` exist. Live probe: `load_state` → `RequestFailedError: Cannot load weights from a sampler weights checkpoint … load_weights only accepts training weights checkpoints`.
- **v3 (`3497a27e … wp-reasoning-v3-final-state`) — loadable, confirmed.** Probe built the client OK: base `Qwen3-30B-A3B`, rank 32. But v3 is the pre-v4 model that **failed REVL-04 codegen** (the reason v4 was created). RL-from-v3 vs v4 baseline risks a gen soft-fail.
- v2 also has a `save_state` checkpoint (`a37be9b1`); not relevant (superseded).

→ To warm-start from the correct (v4) init, **regenerate a v4 `save_state`**: re-run the winning grid cell with `--save-state`. Reproducible (see Execution Plan).

## Finding 3 — THE CONFLICT: v4 MoE-only vs D-09-02 full-MoE RL

| Source | Train flags | Authority |
|---|---|---|
| **v4-winner SFT** | `train_mlp=True, train_attn=False, train_unembed=False` (MoE-only) | D-IT (04.4): *"attention deltas are net-HARMFUL — add codegen damage, slightly hurt judge Spearman, contribute no judge skill"* (STATE.md). Grid driver `_run_grid_train.py` **asserts** `train_attn is False` and aborts otherwise. |
| **RL design D-09-02** | `train_mlp=True, train_attn=True, train_unembed=True` (router frozen) | ROADMAP.md:399 — *"full-MoE RL … attention layers and shared experts"*. |

These are mutually exclusive. **D-09-02 (v2.0-era) was written before the v4 MoE-only finding (04.4-era)** — it is plausibly stale. `create_training_client_from_state` inherits the checkpoint's flags, so a clean warm-start from a (MoE-only) v4 checkpoint *forces* RL to be MoE-only. Honoring D-09-02 instead requires a partial warm-start (load MLP from v4; attn/unembed init fresh) — structurally riskier and re-introduces the attn training v4 deliberately avoided.

---

## Options

### Option A — RL MoE-only (RECOMMENDED). Supersede D-09-02 `train_attn/unembed=True`.
- Regenerate v4 `save_state` (MoE-only), `create_training_client_from_state` → RL trains **MLP/MoE only**, attn+unembed frozen, router frozen.
- **Pros:** (1) LoRA geometry matches the v4 checkpoint exactly — clean, no partial-init risk; (2) consistent with v4's codegen-protective freeze → protects RLEV-01 "no gen regression"; (3) the primary RL target (judge Spearman) lives **entirely in MoE** per D-IT, so MoE-only RL can still move the metric that matters; (4) aligns the RL config with the newer, empirical decision.
- **Cons:** formally contradicts D-09-02 as written → requires an explicit superseding decision (D-09-08?). RL cannot adjust attention (less expressive — but v4 evidence says attn adjustment hurts).

### Option B — RL MLP+attn+unembed (honor D-09-02). Partial warm-start.
- Create RL client with all three flags True; `load_state` the v4 MoE-only weights → MLP warm-starts, attn+unembed start fresh.
- **Pros:** honors D-09-02 full-MoE RL; RL can adapt attention.
- **Cons:** (1) **structure mismatch** — loading a MoE-only checkpoint into a MLP+attn+unembed client is untested (may error or silently leave attn/unembed fresh — needs validation); (2) re-introduces **net-harmful attn deltas** → direct codegen-regression risk against the v4 baseline (the exact failure v4 fixed); (3) highest risk to RLEV-01.

### Option C — RL MoE-only now, revisit attn later (phased).
- Start with Option A. If judge improvement stalls, run a controlled experiment unfreezing attn.
- **Pros:** pragmatic; clean start; keeps Option B in reserve behind evidence.
- **Cons:** defers, not resolves; two potential runs.

---

## Recommendation

**Option A.** The v4 MoE-only finding is newer and empirical; D-09-02's `train_attn=True` predates it and should be superseded. Option A gives a clean warm-start (matching geometry), protects the gen baseline RLEV-01 demands, and still lets RL move the judge (the primary target, which is MoE-borne). Option B's attn training is precisely the codegen-harming lever v4 removed — do not re-introduce it without a specific reason.

If the user prefers to preserve D-09-02's intent, Option C (MoE-only now, attn experiment later behind evidence) is the safe compromise. **Avoid Option B** unless there is a deliberate decision to accept gen-regression risk.

---

## Execution plan (once a train-flag option is signed off)

**Source already decided (2026-06-22): regenerate v4** (Finding 2 settled the source; the only OPEN sign-off is the train-flag option A/B/C). Escape hatch if the source is later reverted to v3: skip Step 1 and set `--init-from tinker://3497a27e-5638-5ac7-97c6-a886062666d9:train:0/weights/wp-reasoning-v3-final-state` (loadable, confirmed; rank-32, MoE-flags as stored) — accept the v3-vs-v4-baseline gen-regression risk noted in Finding 2.

All commands are copy-paste-ready from repo root `/home/robert_li/Desktop/projects/wp-finetune`. Every run sources `.env` for `TINKER_API_KEY` first: `set -a; . ./.env; set +a`. Verified facts that make this self-contained: SFT LR is auto-derived (`hp.get_lr(BASE_MODEL, is_lora=True)`, deterministic — no `--lr` flag); MoE-only is the **default** (`--train-attn`/`--train-unembed` are `store_true default=False`, `train_mlp` always True); `--save-state` writes `tc.save_state(name="{save_name}-final-state")` and records the resulting `state_path` in the manifest JSON.

### Assuming Option A (MoE-only) — the recommended path

**Step 1 — Regenerate the v4 winner with a durable training checkpoint.**
Reproduces the exact winning grid cell (`_run_grid_train.py` passed `--rank 32 --train-path <replay30> --epochs 3 --per-epoch-eval-n 8 --manifest …`; batch-size default 8; MoE-only default), adding `--save-state`:
```bash
set -a; . ./.env; set +a
.venv-tinker/bin/python scripts/tinker_reasoning_sft.py \
  --stage v4-r32-rp30-savestate \
  --rank 32 \
  --train-path data/reasoning_dataset/openai_train.augmented.replay30.jsonl \
  --epochs 3 --batch-size 8 --per-epoch-eval-n 8 \
  --save-state \
  --save-name wp-reasoning-v4-r32-rp30-savestate \
  --manifest output/tinker/wp-reasoning-v4-r32-rp30-savestate-manifest.json
```
- Do NOT pass `--train-attn`/`--train-unembed` → stays MoE-only (matches v4). Confirm the new manifest shows `"train_attn": false, "train_unembed": false`.
- **Equivalence, not byte-identity:** SFT seed determinism is not verified, so the regen reproduces the v4 *recipe*, not necessarily identical weights. The script runs per-epoch eval + its own FS/quality gates — confirm they pass at the same level as the original v4 cell (`04.3-04-SUMMARY.md` / the original `wp-reasoning-v4-r32-rp30-manifest.json` eval numbers). Treat as a valid v1.2-quality init when the gates clear; if you require the *canonical* weights specifically, that is NOT achievable via re-run (the original run only saved sampler_weights) — flag back before proceeding.

**Step 2 — Get the warm-start path.** Read it from the new manifest (do not hand-guess the run id):
```bash
python3 -c "import json;print(json.load(open('output/tinker/wp-reasoning-v4-r32-rp30-savestate-manifest.json'))['state_path'])"
# -> tinker://<NEW_RUN_ID>:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state
```

**Step 3 — Serve the judge (RL prereq).** RL needs `wp_judge` reachable at `http://localhost:8000/v1`:
```bash
bash scripts/serve_v4_judge_vllm.sh   # wait until: curl -s http://localhost:8000/v1/models | grep -q wp_judge
```

**Step 4 — Relaunch RL, warm-started.** Full command (identical to the killed run + `--init-from`; detached). With Option A the checkpoint forces MoE-only via `create_training_client_from_state`, so `--lora-rank`/`--lora-seed` are inert (rank/flags derive from the checkpoint) — kept only for arg-parity:
```bash
set -a; . ./.env; set +a
INIT=$(python3 -c "import json;print(json.load(open('output/tinker/wp-reasoning-v4-r32-rp30-savestate-manifest.json'))['state_path'])")
nohup setsid env WP_JUDGE_DEBUG_DUMP=output/rl_checkpoints/judge_failures.jsonl \
  .venv-tinker/bin/python scripts/rl_train.py \
  --model-id Qwen/Qwen3-30B-A3B --lora-rank 32 --lora-seed 42 \
  --init-from "$INIT" \
  --total-steps 500 --batch-size 8 --checkpoint-every 50 --jaccard-every 20 \
  --kl-soft 0.1 --kl-hard 0.3 --efrac-soft 0.7 --efrac-hard 0.5 \
  --judge-base-url http://localhost:8000/v1 --judge-model wp_judge \
  > output/rl_checkpoints/full_run_warmstart.log 2>&1 < /dev/null &
disown
```
Startup must log `WARM START from tinker://… base_model=Qwen/Qwen3-30B-A3B rank=32 (train_mlp=True attn=False unembed=False)`. If it logs `COLD START`, `--init-from` didn't parse — STOP.

**Step 5 — HARD step-0 validation gate (before letting 500 steps run).** Read `output/rl_checkpoints/full_run_warmstart.log` + the first `rl_metrics.jsonl` line. PASS criteria:
- step-0 `reward_mean` materially **above the cold-start band** (cold-start sat at ~0.20–0.34; a v1.2 init should be clearly higher — expect ≳0.5, calibrate against the v4 SFT's own eval reward if available).
- new entries in `judge_failures.jsonl` ≈ 0 and contain **no** instruct-refusal phrasing (`"I can't assist"`, `"Could you clarify"`); rollouts emit `<?php`/`<wp_gen>` code and `<wp_judge>` rubrics.
- If rollouts still look like raw-base refusals → warm-start did not take; KILL, do not burn the run.

**Step 6 — Re-arm the crash/halt Monitor** (persistent), pointed at the new PID + `full_run_warmstart.log`, filter `Traceback|Error|FAILED|Killed|OOM|halt|RuntimeError|Exception|CUDA|nan` + `PROCESS_DEAD`; resume the monitoring loop.

**Step 7 — Commit** the `--init-from` wiring (`scripts/rl_train.py`, staged-uncommitted) once Step 5 passes end-to-end. Suggested: `fix(09): warm-start RL from v1.2 SFT via --init-from (create_training_client_from_state)`.

### If Option B (MLP+attn+unembed) is chosen instead
Insert between Steps 2 and 4: the warm-start cannot use `create_training_client_from_state` (it would inherit MoE-only). Instead create the client with `create_lora_training_client(train_mlp=True, train_attn=True, train_unembed=True)` then `tc.load_state(<v4 weights path>)`, and **first validate** in a standalone probe that loading a MoE-only checkpoint into a 3-flag client (a) succeeds without shape error and (b) leaves attn/unembed fresh as intended. Add a per-eval codegen-regression watch (attn training is the lever v4 removed for codegen). This requires a small code change to `build_training_client` beyond the current `--init-from`.

### If Option C (MoE-only now, attn later)
Identical to Option A for this run. Defer any attn experiment to a separate, evidence-gated run.

---

## Known-expected behaviors on the warm-started relaunch (do NOT re-investigate)

These were diagnosed this session and are correct-by-design — a future reader should not mistake them for new bugs:

- **`kl_sample_train_v1 = 0.0000` every step** — on-policy artifact (sample + train logprobs from the same within-step weights → IS ratio ≡ 1). GSPO gradient still flows. Not frozen weights.
- **`judge parse failure rate …%` warnings** — DECOUPLED from reward correctness since the non-code guard (`39f10c7`): non-parseable completions get `scalar=0.0` (gen) / `fix_correctness=0.0` (judge), not group-mean imputation. The warning fires inside `compute_group_rewards` but the reward is correct. Watch the *trend*, not the per-step rate.
- **`judge_consistency = 0.0` in Panickssery logs** — genuine (verified: the consistency scorer returns 1.0/0.0/0.4 on known consistent/inconsistent/partial pairs). The base/early policy emits critiques inconsistent with the code. Should shrink as RL trains. Panickssery is logging-only (D-09-05 R1) — no halt.
- **`jaccard_protected = 0.0` (and `None` in JSON)** — monitor-only (frozen router); metric reliability is suspect (None-vs-0.0 mismatch). Informational, not a gate.
- **A non-zero residual judge-parse rate (~3–7%)** is the base-policy prose-not-code tail; expected to *decline* with a v1.2 warm-start (v1.2 emits code/rubrics natively). A sharp *rise* would be the real flag.

## Updating D-09-02 if Option A/C is chosen

The change is a one-line train-flag policy flip + a recorded decision. Files to edit:
- `ROADMAP.md:399` — change `train_attn=True, train_unembed=True` to MoE-only (`train_mlp=True; train_attn=False; train_unembed=False`); `ROADMAP.md:389,394` prose ("full-MoE RL … attention layers") to "MoE-only RL (attn/unembed frozen per D-IT codegen finding)".
- `PROJECT.md` Key Decisions — add e.g. **D-09-08**: "RL trains MoE-only (attn/unembed frozen), superseding D-09-02's `train_attn/unembed=True`; rationale: D-IT showed attn deltas net-harmful to codegen + judge skill is MoE-borne; protects RLEV-01 no-gen-regression."
- `STATE.md` decisions list — mirror the D-09-08 entry.
No code change needed for Option A beyond the already-written `--init-from` (the MoE-only flags come from the checkpoint).

---

## Evidence & citations

- Cold-start code: `scripts/rl_train.py` `build_training_client` / `create_lora_training_client` (pre-fix: no `load_state`).
- Tinker API: `tinker/lib/public_interfaces/service_client.py:245` `create_training_client_from_state`; `training_client.py:721` `load_state` (docstring path `…/weights/…`).
- v4 not loadable: live `load_state` error on `…/sampler_weights/wp-reasoning-v4-r32-rp30-ep3`; manifest `output/tinker/wp-reasoning-v4-r32-rp30-manifest.json` (`state_path: null`, `train_attn:false`, `train_unembed:false`).
- v3 loadable: probe on `tinker://3497a27e-…/weights/wp-reasoning-v3-final-state` (rank 32, base Qwen3-30B-A3B).
- D-09-02: `ROADMAP.md:389,394,399`. v4 MoE-only rationale (D-IT): `STATE.md` "RC-B … attention deltas are net-HARMFUL"; assertion in `scripts/_run_grid_train.py:41-46`.
- RLEV-01/02: `REQUIREMENTS.md:172-173`; Phase 10 baseline: `10-CONTEXT.md:9`.
- Related fixes shipped this session (prerequisite, already committed): parser `8e99619`, extract `d2773e2`, non-code guard `39f10c7`, journal `0e770b9`. `--init-from` wiring: staged, uncommitted.

---

## Sign-off

- [x] **Train-flag option:** A (MoE-only, supersede D-09-02) ☑ / B (MLP+attn+unembed) ☐ / C (MoE-only now, attn later) ☐
- [x] Approve v4 `save_state` regeneration (compute cost: one r32-rp30 SFT, 3 epochs)
- [x] If a decision supersedes D-09-02, record as a new decision ID (**D-09-08**) in PROJECT.md / ROADMAP / STATE.md
- [x] Approver: Dr. Robert Li — date: 2026-06-22T16:47:00+10:00

_Until signed off, the Phase 9 RL run stays down. No compute is burning._
