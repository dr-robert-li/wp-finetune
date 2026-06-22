# Phase 09 — Local-vLLM Consistency RL: Live Status Updates

**Purpose:** remote-watchable status feed for the full 500-step GSPO RL run using the
**$0 local consistency judge** (Option 1). Appended automatically by the monitoring loop.
Newest entries at the BOTTOM of each section.

**Owner:** Dr. Robert Li · **Branch:** `phase10-execution` · **Started:** 2026-06-22

---

## Configuration (locked)

| Item | Value |
|---|---|
| Consistency backend | LOCAL vLLM (`--consistency-base-url http://localhost:8001/v1`) — $0 |
| Consistency model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` (text-only, served `wp_consistency`) |
| Fix-scoring judge | `wp_judge` (v4 merge), vLLM :8000 |
| Consistency weight | 0.3 (D-09-05); fix_correctness 0.7 (deterministic, $0) |
| Warm-start | MoE-only (D-09-08), v4 save_state |
| Serve script | `scripts/serve_consistency_vllm.sh` |

**Model deviation note:** 09-HANDOFF named the *Omni* (vision) NVFP4 variant; the
text-only sibling is used instead (no vision tower, lighter on shared GB10 memory,
identical text-judging quality). User-confirmed 2026-06-22.

---

## Section A — Setup / Download / Serve

- (entries appended below)

---

## Section B — Smoke Test + Throughput (NVFP4 on DGX Spark)

- (entries appended below)

---

## Section C — Dry Run / Signal Check

- (entries appended below)

---

## Section D — Full 500-Step Run: Periodic Status

Each entry: step, reward stats (min/mean/max, non-uniform?), fix_correctness, KL,
checkpoint state, and a **judge-quality observation** (is the local consistency judge
producing sane, non-degenerate scores vs. fix_correctness?).

- (entries appended below)

---

## Section E — Judge Output Quality Audit

Direct spot-checks of local consistency-judge outputs (sampled php/critique pairs →
score), to confirm the local judge is not materially worse than the prior Claude path.

- (entries appended below)

### A1 · 2026-06-22 23:20 — Setup
- Wiring complete: `rl_judge_dispatch.score_judge_consistency_batch` now accepts `base_url`;
  local vLLM path added (`_score_via_vllm` + robust `_parse_consistency_score`). Legacy
  claude path preserved (base_url=None). Offline unit smoke: parser 6/6, batch routing 2/2,
  legacy path intact — ALL PASS.
- `--consistency-base-url` flag added to `rl_train.py`; `rl_rollouts.collect_rollouts` passes it through.
- vLLM image gate (nightly 0.20.2rc1): `NemotronHForCausalLM` registered; all recipe flags
  (`--moe-backend flashinfer_cutlass`, `--kv-cache-dtype fp8`, etc.) valid. `--reasoning-parser`
  omitted by design (thinking disabled at call time).
- Serve script written: `scripts/serve_consistency_vllm.sh` (:8001, served `wp_consistency`, gpu-util 0.30).
- Download started: text-only NVFP4 (19.4GB) via hf_transfer @ ~16MB/s, ETA ~20min.

### A2 · 2026-06-22 23:24 — Download path fix
- First attempts STALLED: repo is Xet-backed; hf_xet CAS connections kept failing
  ("success ratio below threshold (connection struggling)"), rchar delta 0/5s, 0 bytes to disk.
- Fix: `HF_HUB_DISABLE_XET=1` → classic CDN download. Now 14MB/s, ETA ~20min. Waiter armed.

### C1 · 2026-06-22 23:24 — Dry run (mock client)
- `rl_train.py --dry-run --consistency-base-url http://localhost:8001/v1 --consistency-model wp_consistency`:
  PASS. Synthetic GSPO step ran, metrics written (reward_mean=0.9, non-uniform 0.8/1.0,
  kl_v1=0.02), halt_reason=None. New flag accepted; plumbing intact.
- wp_judge :8000 started (loading, ~8min). Consistency :8001 pending download completion.

### A3 · 2026-06-22 23:33 — Memory hygiene
- Stopped idle `unsloth-headless` (3wk idle, 0 VRAM, but had --gpus all + restart=unless-stopped:
  a mid-run VRAM-grab/OOM risk). Manual stop survives daemon restart.
- Now only wp_judge :8000 resident (~69GB VRAM at util 0.55). ~52GB free for consistency :8001
  (needs ~38GB at 0.30). wp_judge confirmed loaded (EngineCore up).

### B1 · 2026-06-22 ~23:58 — Smoke test + NVFP4 throughput (PASS)
- Served `wp_consistency` NVFP4 (flashinfer_cutlass MoE), max_model_len bumped 8192→12288
  after worst-case check showed only 1367 headroom at 8192 (truncation directive).
- Throughput: warm single-call 0.345s; steady decode 54.6 tps single-stream; warm batch-8
  (max-num-seqs 8) = 0.9s wall for 8 concurrent. First batch-8 was 13.3s = one-time cudagraph
  capture; rounds 1-2 stable at 0.9s. Per-RL-step consistency cost ~0.9s (negligible vs ~7min/step).
- Truncation @12288: pathological php+critique = 6221 tok total, headroom 6067, finish=stop. SAFE.

### E1 · 2026-06-22 ~23:58 — Judge-quality (local NVFP4 vs intent)
- Discrimination correct: good critique → 0.8, fabricated "perfectly safe" critique → 0.0
  (good >= wrong), across repeated calls. Clean JSON `{"consistency_score": x}`, no <think> leakage
  with enable_thinking=False. Local judge is NOT degenerate; behaves as a sane consistency rater.

### C2 · 2026-06-23 00:08 — Live signal run, step-0 gates GREEN
- Warm-start gate: `train_mlp=True attn=False unembed=False` (MoE-only, not cold). Pools gen=68 judge=482.
- Step-0 metrics: reward_mean=0.190, min=0.0 max=1.0 → NON-UNIFORM (healthy gradient). n=30. halt=null.
- Local consistency endpoint HIT: vllm :8001 request delta=+20 during the step → $0 path live, no claude.
- MemAvailable 13.4GB stable (OOM guard armed <2GB). 12-step signal run continuing.

### D · 2026-06-22 14:08 UTC — RL status tick
- containers: wp_judge=`Up 44 minutes` | wp_consistency=`Up 15 minutes`
- metrics: step=0 reward_mean=0.19038666666666668 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None 
- warm_start: 2026-06-22 23:58:52,609 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 14:08 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique
