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

### D · 2026-06-23 00:30 — FULL 500-STEP RUN LAUNCHED
- Warm-start MoE-only gate GREEN. Canonical metrics=rl_metrics.jsonl, checkpoint-every 50.
- Consistency=local :8001 ($0). Judge=:8000. OOM guard armed (<2048MB). pid 1540083.
- Step cadence ~8.5min → ETA ~3 days. Monitoring loop ticks every 20min below.

### D · 2026-06-22 14:30 UTC — RL status tick
- containers: wp_judge=`Up About an hour` | wp_consistency=`Up 38 minutes`
- metrics: NO METRICS YET (run not started or first step pending)
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 14:30 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-22 14:32 UTC — RL status tick
- containers: wp_judge=`Up About an hour` | wp_consistency=`Up 40 minutes`
- metrics: NO METRICS YET (run not started or first step pending)
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 14:32 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 00:32 — Run healthy, monitoring hardened
- Full run pid 1540083 ALIVE, warm-started, ~2min in (step 0 ~8min/step → first metrics ~00:38). "No
  metrics yet" earlier was expected, not a failure.
- Monitoring liveness switched to pidfile `kill -0` (robust). Doc-tick loop every 20min. OOM guard <2048MB.
- Watchers: full-run task (exit notify) + OOM guard + doc-loop. Self-checks every ~25min.

### D · 2026-06-22 14:52 UTC — RL status tick
- containers: wp_judge=`Up About an hour` | wp_consistency=`Up About an hour`
- metrics: step=1 reward_mean=0.22874999999999998 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.329→0.229 over 2 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 14:52 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.2 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 00:58 — self-check (healthy)
- 3 steps: reward 0.329/0.229/0.268, non-uniform every step (min0/max1), no collapse. 0 errors, halt=None.
- ~8min/step → 500 ETA ~2.7 days. Mem 12.7GB stable. Judge-quality good1.0>=wrong0.2 OK.
- Too early for trend; signal alive + healthy. Next: watch step-50 checkpoint write (~6h).

### D · 2026-06-22 15:12 UTC — RL status tick
- containers: wp_judge=`Up 2 hours` | wp_consistency=`Up About an hour`
- metrics: step=4 reward_mean=0.2849354838709677 min=0.0 max=0.9964999999999999 non_uniform=True kl_v1=0.0 halt=None trend 0.329→0.285 over 5 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 15:12 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 01:26 — self-check (healthy, 7 steps)
- reward mean[0.177,0.329], non-uniform every step, no collapse. Noisy ~0.2-0.3 (matches prior run's
  early flat; trend not yet readable). 0 errors. Mem 12.6GB. Judge good1.0>=wrong0.0 OK. ckpt@50 ~5.5h.

### D · 2026-06-22 15:32 UTC — RL status tick
- containers: wp_judge=`Up 2 hours` | wp_consistency=`Up 2 hours`
- metrics: step=7 reward_mean=0.21769687499999998 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.268→0.218 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 15:32 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-22 15:52 UTC — RL status tick
- containers: wp_judge=`Up 2 hours` | wp_consistency=`Up 2 hours`
- metrics: step=9 reward_mean=0.224475 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.285→0.224 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 15:52 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 01:55 — self-check (healthy, 11 steps)
- reward all non-uniform, mean[0.177,0.329]. recent5=0.228 vs early5=0.276 — flat, within noise (expected
  this early per handoff; revisit only if still flat past ~100 steps). 0 errors. Mem 12.6GB. Judge OK. ckpt@50 ~5h.

### D · 2026-06-22 16:12 UTC — RL status tick
- containers: wp_judge=`Up 3 hours` | wp_consistency=`Up 2 hours`
- metrics: step=12 reward_mean=0.4778 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.218→0.478 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 16:12 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 02:23 — self-check (healthy, 15 steps, early uptick)
- reward[0.177,0.478] all non-uniform. recent5=0.336 > first5=0.276, peaks 0.43-0.48 → tentative rise vs
  prior run's flat ~0.2 (still noisy, not conclusive). 0 errors. Mem 12.4GB. Judge OK. ckpt@50 ~4.5h.

### D · 2026-06-22 16:32 UTC — RL status tick
- containers: wp_judge=`Up 3 hours` | wp_consistency=`Up 3 hours`
- metrics: step=15 reward_mean=0.23971562500000002 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.259→0.240 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 16:32 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 02:52 — self-check (healthy, 18 steps)
- reward oscillating [0.126,0.478], recent5=0.258 ~ first5=0.276 (flat; step12-14 uptick was noise). All
  non-uniform, no halts. 0 errors. Mem 12.4GB. Judge OK. Trend still unreadable at 18 steps. ckpt@50 ~4h.

### D · 2026-06-22 16:52 UTC — RL status tick
- containers: wp_judge=`Up 3 hours` | wp_consistency=`Up 3 hours`
- metrics: step=17 reward_mean=0.22585937499999997 min=0.0 max=0.9964999999999999 non_uniform=True kl_v1=0.0 halt=None trend 0.478→0.226 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 16:52 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-22 17:12 UTC — RL status tick
- containers: wp_judge=`Up 4 hours` | wp_consistency=`Up 3 hours`
- metrics: step=20 reward_mean=0.12813870967741936 min=0.0 max=0.8335999999999999 non_uniform=True kl_v1=0.0 halt=None trend 0.240→0.128 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 17:12 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 03:23 — self-check (healthy, 22 steps)
- reward [0.126,0.478], recent10=0.269 vs first10=0.246 (noisy-flat, marginal). All non-uniform, no halts.
  0 errors. Mem 12.2GB (slow ~400MB/1.5h drift, far from 2GB floor — watching). Judge OK. ckpt@50 ~3.5h.

### D · 2026-06-22 17:32 UTC — RL status tick
- containers: wp_judge=`Up 4 hours` | wp_consistency=`Up 4 hours`
- metrics: step=23 reward_mean=0.23313124999999998 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.300→0.233 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 17:32 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-22 17:52 UTC — RL status tick
- containers: wp_judge=`Up 4 hours` | wp_consistency=`Up 4 hours`
- metrics: step=25 reward_mean=0.1411625 min=0.0 max=0.9371999999999999 non_uniform=True kl_v1=0.0 halt=None trend 0.128→0.141 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 17:52 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 03:54 — self-check (healthy, 26 steps)
- reward [0.126,0.478] flat-noisy (recent10=0.229 ~ first10=0.246). All non-uniform, no halts. 0 errors.
- Mem 12.4GB (oscillating 12.2-12.6, NOT monotonic — no leak). Judge OK. ckpt@50 ~3h.

### D · 2026-06-22 18:12 UTC — RL status tick
- containers: wp_judge=`Up 5 hours` | wp_consistency=`Up 4 hours`
- metrics: step=28 reward_mean=0.31580625 min=0.0 max=0.9964999999999999 non_uniform=True kl_v1=0.0 halt=None trend 0.233→0.316 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 18:12 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 04:25 — self-check (healthy, 30 steps)
- reward flat: recent10=0.244 ~ first10=0.246, range [0.126,0.478]. All non-uniform, no halts. 0 errors.
- Mem 12.3GB stable. Judge OK. ckpt@50 ~2.5h. Flat is fine <100 steps (watch-item threshold).

### D · 2026-06-22 18:32 UTC — RL status tick
- containers: wp_judge=`Up 5 hours` | wp_consistency=`Up 5 hours`
- metrics: step=30 reward_mean=0.316546875 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.141→0.317 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 18:32 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-22 18:52 UTC — RL status tick
- containers: wp_judge=`Up 5 hours` | wp_consistency=`Up 5 hours`
- metrics: step=33 reward_mean=0.15915625 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.316→0.159 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 18:52 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 04:56 — self-check (healthy, 35 steps)
- reward flat-noisy (recent10=0.258 ~ first10=0.246). All non-uniform, no halts. 0 errors. Mem 12.2GB. Judge OK. ckpt@50 ~2h.

### D · 2026-06-22 19:12 UTC — RL status tick
- containers: wp_judge=`Up 6 hours` | wp_consistency=`Up 5 hours`
- metrics: step=36 reward_mean=0.20993437499999995 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.362→0.210 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 19:12 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 05:27 — self-check (healthy, 39 steps)
- reward flat-noisy (recent10=0.266 vs first10=0.246). All non-uniform, no halts. 0 errors. Mem 12.3GB. Judge OK. ckpt@50 ~11 steps out.

### D · 2026-06-22 19:32 UTC — RL status tick
- containers: wp_judge=`Up 6 hours` | wp_consistency=`Up 6 hours`
- metrics: step=38 reward_mean=0.292875 min=0.0 max=0.9379 non_uniform=True kl_v1=0.0 halt=None trend 0.159→0.293 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 19:32 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-22 19:52 UTC — RL status tick
- containers: wp_judge=`Up 6 hours` | wp_consistency=`Up 6 hours`
- metrics: step=41 reward_mean=0.3333451612903226 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.210→0.333 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 19:52 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 05:58 — self-check (healthy, 43 steps)
- reward flat (recent10=0.263 vs first10=0.246). All non-uniform, no halts. 0 errors. Mem 12.1GB. ckpt@50 ~7 steps out.

### D · 2026-06-22 20:12 UTC — RL status tick
- containers: wp_judge=`Up 7 hours` | wp_consistency=`Up 6 hours`
- metrics: step=44 reward_mean=0.29213958333333334 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.264→0.292 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 20:12 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-22 20:33 UTC — RL status tick
- containers: wp_judge=`Up 7 hours` | wp_consistency=`Up 7 hours`
- metrics: step=46 reward_mean=0.21187499999999998 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.333→0.212 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 20:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 06:33 — self-check (healthy, 47 steps)
- reward recent10=0.290 vs first10=0.246 (mild uptick, within noise). All non-uniform, no halts. 0 errors.
- Mem 12.2GB. ckpt@50 ~4 steps out (verify write next check).

### D · 2026-06-22 20:53 UTC — RL status tick
- containers: wp_judge=`Up 7 hours` | wp_consistency=`Up 7 hours`
- metrics: step=49 reward_mean=0.343284375 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.292→0.343 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 20:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 07:06 — MILESTONE: step-50 checkpoint WROTE ✅
- checkpoint_manifest.json now has step-50 entry (sampler_weights/step-50, saved 20:51Z). Handoff's
  "empty checkpoints" concern debunked — write path works (prior run just died pre-50).
- reward recent10=0.286 vs first10=0.246 (mild uptick holding). All non-uniform, no halts. 0 errors. Mem 12.2GB.

### D · 2026-06-22 21:13 UTC — RL status tick
- containers: wp_judge=`Up 8 hours` | wp_consistency=`Up 7 hours`
- metrics: step=51 reward_mean=0.19404375 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.212→0.194 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 21:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-22 21:33 UTC — RL status tick
- containers: wp_judge=`Up 8 hours` | wp_consistency=`Up 8 hours`
- metrics: step=54 reward_mean=0.1740875 min=0.0 max=0.9371999999999999 non_uniform=True kl_v1=0.0 halt=None trend 0.343→0.174 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 21:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 07:42 — self-check (healthy, 56 steps)
- reward flat: recent10=0.246 = first10=0.246 (prior upticks were noise; ~0.25 over 56 steps). Non-uniform,
  no halts. 0 errors. Mem 11.9GB (band 11.9-12.6, no leak). Judge OK. 1 ckpt. Watch flat vs 100-step threshold.

---

## Section F — KL=0.0 Watch-Item + Decision Points (added 2026-06-23 07:50, user-requested)

**What kl_sample_train_v1 actually is:** sampling-vs-training logprob divergence WITHIN a step
(an off-policy/staleness tripwire feeding the GRPO-08 autohalt: kl_v1>kl_hard 0.3 → HARD halt).
It is NOT a KL-penalty-to-reference (warm-start) term. Source: rl_train._compute_kl_metrics →
tinker_cookbook.rl.metrics.compute_kl_sample_train(data, training_lps).

**Genuine-0, not a bug (verified at step 56):** _compute_kl_metrics returns the 0.0 FALLBACK only
when data/training_lps are empty. But e_max_violation_mean (~6.5) and e_frac_with_tokens_mean (~0.95)
are non-zero and moving every step — these derive from the SAME training logprobs the fallback would
have zeroed. So training_lps ARE present → kl_v1=0.0 is the real compute_kl_sample_train result
(consistent with the handoff "on-policy artifact": sample and train share the in-step weight snapshot).

**The real policy-drift constraint is GSPO ratio clipping, and it IS active:** e_frac_with_tokens_mean
~0.95 (vs --efrac-soft 0.7 / --efrac-hard 0.5) and e_max_violation ~6.5 are the sequence/token
importance-ratio guards. Drift is therefore constrained even though the kl_sample_train staleness
metric reads 0 — i.e. "missing KL penalty → unconstrained drift → reward hacking" is mitigated by the
efrac/e_max mechanism, not by kl_sample_train.

**Open nuance to reconcile at step 100:** token-level e_max_violation ~6.5 shows a real sample↔train
gap, yet sequence-level kl_v1 aggregates to exactly 0. If kl_v1 is STILL exactly 0.0 at step 100, confirm
(a) it's genuine on-policy (not a logging-path bug), and (b) the efrac/e_max guard remains the active
drift constraint. Do NOT change config mid-run; surface to user if anomalous.

### Decision points (track here)
| Milestone | ETA | Check |
|---|---|---|
| Step 100 | ~6h | reward_mean trending up from ~0.25? kl_v1 > 0 or still genuine-0? mean[51-100] vs mean[0-50]? |
| Step 100 ckpt | ~6h | verify write to checkpoint_manifest.json |
| Step 150-200 | ~12h | first real signal window — sustained flat here would be concerning (revisit LR/GSPO w/ user) |

### D · 2026-06-22 21:53 UTC — RL status tick
- containers: wp_judge=`Up 8 hours` | wp_consistency=`Up 8 hours`
- metrics: step=56 reward_mean=0.2875550986842105 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.194→0.288 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 21:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### F-CORRECTION · 2026-06-23 08:00 — efrac directionality (read from check_halt, supersedes Section F mislabel)
Verified against rl_train.check_halt — correcting my earlier Section F wording:
- **e_frac_with_tokens_mean = MoE ROUTING health** (fraction of tokens routed to any expert), NOT the
  importance-sampling ratio guard I mislabeled it as. **HIGHER = BETTER.**
  Halt logic: `e_frac < efrac_hard 0.5` → HARD halt (router collapse); `< efrac_soft 0.7` → soft alert.
- **Directionality (answers the watch):** current e_frac ~0.95 is ABOVE soft 0.7, healthy. The concerning
  direction is **DOWNWARD toward hard 0.5** (NOT upward). So 0.95 is a comfortable ~0.45 margin, not "above
  soft in a concerning direction." The real pre-halt signal = e_frac TRENDING DOWN toward 0.5.
- **Trend so far (56 steps):** e_frac first10 mean 0.960 → recent10 0.959 = FLAT. Band 0.943-0.986. No
  downward drift → no pre-halt signal. Will track for a decreasing trend over the next 44 steps.
- **e_max_violation (~6.3, band 5.75-6.98) = MoE expert load-balance/capacity diagnostic. LOGGED ONLY, NOT a
  halt guard** (not referenced in check_halt). Stable, not trending.
- **Net: the two actual halt guards are kl_v1>0.3 (currently 0, on-policy) and e_frac<0.5 (currently 0.95,
  flat). e_frac directionality is the better pre-halt signal than kl_v1 — and it is currently flat/healthy.**

### D · 2026-06-22 22:13 UTC — RL status tick
- containers: wp_judge=`Up 9 hours` | wp_consistency=`Up 8 hours`
- metrics: step=59 reward_mean=0.175525 min=0.0 max=0.9323 non_uniform=True kl_v1=0.0 halt=None trend 0.174→0.176 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 22:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 08:31 — self-check (healthy, 63 steps)
- reward recent10=0.261 vs first10=0.246 (flat, non-uniform, no halts). kl_v1 genuine-0.
- e_frac recent10=0.961 vs first10=0.960 FLAT (margin ~0.46 to halt 0.5; no downward trend = no pre-halt). e_max ~6.4 stable.
- 0 errors. Mem 11.7GB (band 11.7-12.6, no leak). Judge OK. Next decision pt: step 100 (~4.5h).

### D · 2026-06-22 22:33 UTC — RL status tick
- containers: wp_judge=`Up 9 hours` | wp_consistency=`Up 9 hours`
- metrics: step=62 reward_mean=0.247928125 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.250→0.248 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 22:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-22 22:53 UTC — RL status tick
- containers: wp_judge=`Up 9 hours` | wp_consistency=`Up 9 hours`
- metrics: step=64 reward_mean=0.34360625 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.176→0.344 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 22:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 09:07 — self-check (healthy, 67 steps) + MEM WATCH
- reward recent10=0.256 vs first10=0.246 (flat). kl_v1=0. e_frac recent10=0.964 vs first10=0.960 (flat/up, safe).
  e_max 6.43. 0 errors. Judge OK. Non-uniform, no halts.
- ⚠ MEM WATCH: MemAvailable 11.3GB, slow decline 12.6→11.3 over ~3.5h (~370MB/h). Not urgent (5.6x floor)
  but tracking slope; ~58h run left. OOM guard (<2GB) is the backstop. Decomposing anon-vs-cache to ID source.

### F-MEM · 2026-06-23 09:08 — memory decomposition (baseline for leak-tracking)
- MemTotal 122GB, MemFree 2.49GB, MemAvailable 11.2GB (incl 8.6GB reclaimable Cached), AnonPages 9.9GB.
- Bulk of the 122GB is GPU/unified model weights (judge ~69GB + consistency ~25GB VRAM); container CPU-RSS
  is small (judge 1.5GB, consistency 2.6GB). System is legitimately near-full from two 30B models.
- The MemAvailable "decline" is partly cache flux (MemAvailable counts cache as available, so cache growth
  doesn't lower it; AnonPages growth does). Now tracking ANONPAGES + MEMFREE each tick: a steadily rising
  AnonPages = a real process leak → will flag to user. OOM guard (MemAvailable<2048MB) remains the backstop;
  standard Linux reclaims the 8.6GB cache synchronously under allocation, covering the MemFree-vs-Available gap.
- Baseline to watch: AnonPages 9883MB, MemFree 2491MB @ step 66.

### D · 2026-06-22 23:13 UTC — RL status tick
- containers: wp_judge=`Up 10 hours` | wp_consistency=`Up 9 hours`
- metrics: step=67 reward_mean=0.3586636513157895 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.248→0.359 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 23:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-22 23:33 UTC — RL status tick
- containers: wp_judge=`Up 10 hours` | wp_consistency=`Up 10 hours`
- metrics: step=70 reward_mean=0.24042582236842105 min=0.0 max=0.9365 non_uniform=True kl_v1=0.0 halt=None trend 0.189→0.240 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 23:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 09:37 — self-check (healthy, 70 steps) — MEM LEAK RULED OUT
- reward recent10=0.263 vs first10=0.246 (flat). kl_v1=0. e_frac recent10=0.958 (flat, safe). e_max 6.43. 0 errors. Non-uniform, no halts.
- MEM: AnonPages 9767MB (DOWN from 9883 baseline) + MemFree 3648MB (UP from 2491) → NO leak; prior
  MemAvailable dip was cache flux. Memory healthy, downgrading mem-watch to routine.

### D · 2026-06-22 23:53 UTC — RL status tick
- containers: wp_judge=`Up 10 hours` | wp_consistency=`Up 10 hours`
- metrics: step=72 reward_mean=0.2067 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.359→0.207 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-22 23:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 00:13 UTC — RL status tick
- containers: wp_judge=`Up 11 hours` | wp_consistency=`Up 10 hours`
- metrics: step=74 reward_mean=0.23611875 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.306→0.236 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 00:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 00:33 UTC — RL status tick
- containers: wp_judge=`Up 11 hours` | wp_consistency=`Up 11 hours`
- metrics: step=76 reward_mean=0.35665625 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.389→0.357 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 00:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 00:53 UTC — RL status tick
- containers: wp_judge=`Up 11 hours` | wp_consistency=`Up 11 hours`
- metrics: step=79 reward_mean=0.2958774193548387 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.236→0.296 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 00:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 01:13 UTC — RL status tick
- containers: wp_judge=`Up 12 hours` | wp_consistency=`Up 11 hours`
- metrics: step=81 reward_mean=0.27079374999999994 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.357→0.271 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 01:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 01:33 UTC — RL status tick
- containers: wp_judge=`Up 12 hours` | wp_consistency=`Up 12 hours`
- metrics: step=84 reward_mean=0.35810624999999996 min=0.0 max=0.9993000000000001 non_uniform=True kl_v1=0.0 halt=None trend 0.296→0.358 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 01:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 01:53 UTC — RL status tick
- containers: wp_judge=`Up 12 hours` | wp_consistency=`Up 12 hours`
- metrics: step=87 reward_mean=0.30904687499999994 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.268→0.309 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 01:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 02:13 UTC — RL status tick
- containers: wp_judge=`Up 13 hours` | wp_consistency=`Up 12 hours`
- metrics: step=90 reward_mean=0.41996875 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.184→0.420 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 02:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 02:33 UTC — RL status tick
- containers: wp_judge=`Up 13 hours` | wp_consistency=`Up 13 hours`
- metrics: step=92 reward_mean=0.23612903225806448 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.309→0.236 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 02:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 02:53 UTC — RL status tick
- containers: wp_judge=`Up 13 hours` | wp_consistency=`Up 13 hours`
- metrics: step=95 reward_mean=0.241640625 min=0.0 max=0.9637 non_uniform=True kl_v1=0.0 halt=None trend 0.420→0.242 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 02:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 03:13 UTC — RL status tick
- containers: wp_judge=`Up 14 hours` | wp_consistency=`Up 13 hours`
- metrics: step=97 reward_mean=0.24180953947368422 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.236→0.242 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 03:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 03:33 UTC — RL status tick
- containers: wp_judge=`Up 14 hours` | wp_consistency=`Up 14 hours`
- metrics: step=100 reward_mean=0.293259375 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.242→0.293 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 03:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 03:53 UTC — RL status tick
- containers: wp_judge=`Up 14 hours` | wp_consistency=`Up 14 hours`
- metrics: step=102 reward_mean=0.2244483870967742 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.242→0.224 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 03:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 04:13 UTC — RL status tick
- containers: wp_judge=`Up 15 hours` | wp_consistency=`Up 14 hours`
- metrics: step=105 reward_mean=0.247290625 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.293→0.247 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 04:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 04:33 UTC — RL status tick
- containers: wp_judge=`Up 15 hours` | wp_consistency=`Up 15 hours`
- metrics: step=107 reward_mean=0.215584375 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.224→0.216 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 04:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 04:53 UTC — RL status tick
- containers: wp_judge=`Up 15 hours` | wp_consistency=`Up 15 hours`
- metrics: step=110 reward_mean=0.2737830592105263 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.247→0.274 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 04:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 05:13 UTC — RL status tick
- containers: wp_judge=`Up 16 hours` | wp_consistency=`Up 15 hours`
- metrics: step=112 reward_mean=0.25245937499999993 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.216→0.252 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 05:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 05:33 UTC — RL status tick
- containers: wp_judge=`Up 16 hours` | wp_consistency=`Up 16 hours`
- metrics: step=115 reward_mean=0.3841375 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.274→0.384 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 05:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 05:53 UTC — RL status tick
- containers: wp_judge=`Up 16 hours` | wp_consistency=`Up 16 hours`
- metrics: step=117 reward_mean=0.25779687500000004 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.252→0.258 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 05:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 06:13 UTC — RL status tick
- containers: wp_judge=`Up 17 hours` | wp_consistency=`Up 16 hours`
- metrics: step=120 reward_mean=0.23097812499999998 min=0.0 max=0.9379 non_uniform=True kl_v1=0.0 halt=None trend 0.384→0.231 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 06:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### F-STEP100 · 2026-06-23 16:18 — DECISION POINT step 100 reconciled (run at step 122)
- Checkpoints step-50 AND step-100 both WROTE ✓ (2 in manifest).
- Reward windows: mean[0-50]=0.266 → mean[51-100]=0.278 = +0.012 (marginal; essentially flat, not collapsing).
- kl_v1 STILL exactly 0.0 at step 122. Per Section F this is genuine on-policy (NOT a zeroed KL coefficient):
  efrac/e_max derive from live training_lps so data IS present; kl_sample_train measures sample-vs-train
  staleness which is ~0 on-policy. The active drift guard is e_frac (0.957, flat, halt<0.5) — healthy. No config bug.
- VERDICT: reward at the handoff's 100-step flat threshold but only marginally so; per user's plan the
  decisive window is STEP 150-200. e_frac/kl/mem/judge all healthy. ~8min/step, ETA ~2 days. Continue.

### D · 2026-06-23 06:33 UTC — RL status tick
- containers: wp_judge=`Up 17 hours` | wp_consistency=`Up 17 hours`
- metrics: step=123 reward_mean=0.24424374999999998 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.330→0.244 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 06:33 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 06:53 UTC — RL status tick
- containers: wp_judge=`Up 17 hours` | wp_consistency=`Up 17 hours`
- metrics: step=125 reward_mean=0.3749675986842105 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.231→0.375 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 06:53 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 17:06 — self-check (healthy, 127 steps)
- reward windows monotonic creep: [0-50]=0.266 → [51-100]=0.278 → [101-127]=0.280 (tiny but consistent up).
  Non-uniform, no halts. kl_v1=0. e_frac recent10=0.958 vs first10=0.960 (flat, safe). e_max 6.30. 0 errors.
  Mem AnonPages 9688 (no leak). Judge OK. Decisive window 150-200 next.

### D · 2026-06-23 07:13 UTC — RL status tick
- containers: wp_judge=`Up 18 hours` | wp_consistency=`Up 17 hours`
- metrics: step=128 reward_mean=0.21990213815789472 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.244→0.220 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 07:13 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 07:34 UTC — RL status tick
- containers: wp_judge=`Up 18 hours` | wp_consistency=`Up 18 hours`
- metrics: step=130 reward_mean=0.23870357142857143 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.375→0.239 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 07:34 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 17:47 — self-check (healthy, 132 steps)
- reward plateau ~0.277: [0-50]=0.266 [51-100]=0.278 [101-132]=0.277 (creep flattened, holds +0.011 vs early).
  Non-uniform, no halts. kl_v1=0. e_frac 0.960 dead-flat (safe). e_max 6.25. 0 errors. AnonPages 9813 (no leak). Judge OK.
- Decisive 150-200 window next; step-150 ckpt due ~2.5h.

### D · 2026-06-23 07:54 UTC — RL status tick
- containers: wp_judge=`Up 19 hours` | wp_consistency=`Up 18 hours`
- metrics: step=133 reward_mean=0.24034062499999997 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.220→0.240 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 07:54 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 08:14 UTC — RL status tick
- containers: wp_judge=`Up 19 hours` | wp_consistency=`Up 18 hours`
- metrics: step=136 reward_mean=0.29476875 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.207→0.295 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 08:14 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 18:28 — self-check (healthy, 137 steps)
- reward plateau ~0.275 ([0-50]=0.266 [51-100]=0.278 [101-137]=0.274). Non-uniform, no halts. kl_v1=0.
  e_frac 0.960 flat/safe. e_max 6.30. 0 errors. AnonPages 9730 (no leak). Judge OK. Entering 150-200 window soon.

### D · 2026-06-23 08:34 UTC — RL status tick
- containers: wp_judge=`Up 19 hours` | wp_consistency=`Up 19 hours`
- metrics: step=138 reward_mean=0.262775 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.240→0.263 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 08:34 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 08:54 UTC — RL status tick
- containers: wp_judge=`Up 20 hours` | wp_consistency=`Up 19 hours`
- metrics: step=141 reward_mean=0.3117875 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.295→0.312 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 08:54 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 19:13 — self-check (healthy, 143 steps)
- reward flat ~0.273 ([0-50]=0.266 [51-100]=0.278 [101-143]=0.272). Non-uniform, no halts. kl_v1=0.
  e_frac 0.956 flat/safe. e_max 6.39. 0 errors. AnonPages 9688 (no leak). Judge OK. step-150 ckpt ~7 steps out.

### D · 2026-06-23 09:14 UTC — RL status tick
- containers: wp_judge=`Up 20 hours` | wp_consistency=`Up 19 hours`
- metrics: step=144 reward_mean=0.35031428571428563 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.350→0.350 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 09:14 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 09:34 UTC — RL status tick
- containers: wp_judge=`Up 20 hours` | wp_consistency=`Up 20 hours`
- metrics: step=146 reward_mean=0.223059375 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.312→0.223 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 09:34 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.2 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 09:54 UTC — RL status tick
- containers: wp_judge=`Up 21 hours` | wp_consistency=`Up 20 hours`
- metrics: step=149 reward_mean=0.228575 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.350→0.229 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 09:54 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 10:14 UTC — RL status tick
- containers: wp_judge=`Up 21 hours` | wp_consistency=`Up 20 hours`
- metrics: step=151 reward_mean=0.25328125 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.223→0.253 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 10:14 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 19:57 — self-check (healthy, 154 steps) — step-150 ckpt WROTE ✓
- step-150 checkpoint in manifest (3 total: 50/100/150).
- reward flat ~0.275: [0-50]=0.266 [51-100]=0.278 [101-150]=0.274 [150+]=0.278 (only 4 steps into decisive
  150-200 window — too early to call). Non-uniform, no halts. kl_v1=0. e_frac 0.973 flat/safe. e_max 6.57.
  0 errors. AnonPages 9684 (no leak). Judge OK.

### D · 2026-06-23 10:34 UTC — RL status tick
- containers: wp_judge=`Up 21 hours` | wp_consistency=`Up 21 hours`
- metrics: step=154 reward_mean=0.31653125 min=0.0 max=0.9923 non_uniform=True kl_v1=0.0 halt=None trend 0.229→0.317 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 10:34 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 10:54 UTC — RL status tick
- containers: wp_judge=`Up 22 hours` | wp_consistency=`Up 21 hours`
- metrics: step=157 reward_mean=0.2336625 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.313→0.234 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 10:54 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 11:14 UTC — RL status tick
- containers: wp_judge=`Up 22 hours` | wp_consistency=`Up 21 hours`
- metrics: step=159 reward_mean=0.17438749999999997 min=0.0 max=0.9379 non_uniform=True kl_v1=0.0 halt=None trend 0.317→0.174 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 11:14 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 21:16 — self-check (healthy, 159 steps)
- reward [51-100]=0.278 [101-150]=0.274 [150+]=0.259 (10 steps into decisive window; slightly below plateau
  but within noise, not collapsing). Non-uniform, no halts. kl_v1=0. e_frac 0.958 flat/safe. e_max 6.41.
  0 errors. AnonPages 9771 (no leak). Judge OK. Watching window through ~200 for flat-vs-rise decision.

### D · 2026-06-23 11:34 UTC — RL status tick
- containers: wp_judge=`Up 22 hours` | wp_consistency=`Up 22 hours`
- metrics: step=162 reward_mean=0.319128125 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.234→0.319 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 11:34 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 11:54 UTC — RL status tick
- containers: wp_judge=`Up 23 hours` | wp_consistency=`Up 22 hours`
- metrics: step=164 reward_mean=0.21187499999999998 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.174→0.212 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 11:54 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 12:14 UTC — RL status tick
- containers: wp_judge=`Up 23 hours` | wp_consistency=`Up 22 hours`
- metrics: step=167 reward_mean=0.291021875 min=0.0 max=0.9964999999999999 non_uniform=True kl_v1=0.0 halt=None trend 0.319→0.291 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 12:14 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 12:34 UTC — RL status tick
- containers: wp_judge=`Up 23 hours` | wp_consistency=`Up 23 hours`
- metrics: step=170 reward_mean=0.15915 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.291→0.159 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 12:34 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 12:54 UTC — RL status tick
- containers: wp_judge=`Up 24 hours` | wp_consistency=`Up 23 hours`
- metrics: step=172 reward_mean=0.38684999999999997 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.291→0.387 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 12:54 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.2 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 13:14 UTC — RL status tick
- containers: wp_judge=`Up 24 hours` | wp_consistency=`Up 23 hours`
- metrics: step=174 reward_mean=0.293915625 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.213→0.294 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 13:14 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 13:34 UTC — RL status tick
- containers: wp_judge=`Up 24 hours` | wp_consistency=`Up 24 hours`
- metrics: step=177 reward_mean=0.275509375 min=0.0 max=0.9908999999999999 non_uniform=True kl_v1=0.0 halt=None trend 0.387→0.276 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 13:34 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 13:54 UTC — RL status tick
- containers: wp_judge=`Up 25 hours` | wp_consistency=`Up 24 hours`
- metrics: step=180 reward_mean=0.27750312499999996 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.131→0.278 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 13:54 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 14:14 UTC — RL status tick
- containers: wp_judge=`Up 25 hours` | wp_consistency=`Up 24 hours`
- metrics: step=182 reward_mean=0.3477464285714285 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.276→0.348 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 14:14 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 14:34 UTC — RL status tick
- containers: wp_judge=`Up 25 hours` | wp_consistency=`Up 25 hours`
- metrics: step=185 reward_mean=0.18784062499999998 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.278→0.188 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 14:34 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 14:54 UTC — RL status tick
- containers: wp_judge=`Up 26 hours` | wp_consistency=`Up 25 hours`
- metrics: step=188 reward_mean=0.26418437499999997 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.257→0.264 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 14:54 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 15:14 UTC — RL status tick
- containers: wp_judge=`Up 26 hours` | wp_consistency=`Up 25 hours`
- metrics: step=190 reward_mean=0.2340625 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.188→0.234 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 15:14 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 15:34 UTC — RL status tick
- containers: wp_judge=`Up 26 hours` | wp_consistency=`Up 26 hours`
- metrics: step=193 reward_mean=0.217346875 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.264→0.217 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 15:34 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 15:54 UTC — RL status tick
- containers: wp_judge=`Up 27 hours` | wp_consistency=`Up 26 hours`
- metrics: step=196 reward_mean=0.1666217105263158 min=0.0 max=0.9964999999999999 non_uniform=True kl_v1=0.0 halt=None trend 0.188→0.167 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 15:54 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 16:14 UTC — RL status tick
- containers: wp_judge=`Up 27 hours` | wp_consistency=`Up 26 hours`
- metrics: step=198 reward_mean=0.286196875 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.217→0.286 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 16:14 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 16:34 UTC — RL status tick
- containers: wp_judge=`Up 27 hours` | wp_consistency=`Up 27 hours`
- metrics: step=201 reward_mean=0.2332375 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.167→0.233 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 16:34 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 16:54 UTC — RL status tick
- containers: wp_judge=`Up 28 hours` | wp_consistency=`Up 27 hours`
- metrics: step=204 reward_mean=0.26060937500000003 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.363→0.261 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 16:54 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 17:14 UTC — RL status tick
- containers: wp_judge=`Up 28 hours` | wp_consistency=`Up 27 hours`
- metrics: step=206 reward_mean=0.31462812500000004 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.233→0.315 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 17:14 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 17:35 UTC — RL status tick
- containers: wp_judge=`Up 28 hours` | wp_consistency=`Up 28 hours`
- metrics: step=209 reward_mean=0.15716875 min=0.0 max=0.94 non_uniform=True kl_v1=0.0 halt=None trend 0.261→0.157 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 17:35 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 17:55 UTC — RL status tick
- containers: wp_judge=`Up 29 hours` | wp_consistency=`Up 28 hours`
- metrics: step=212 reward_mean=0.33697499999999997 min=0.0 max=0.9944 non_uniform=True kl_v1=0.0 halt=None trend 0.127→0.337 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 17:55 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 18:15 UTC — RL status tick
- containers: wp_judge=`Up 29 hours` | wp_consistency=`Up 28 hours`
- metrics: step=214 reward_mean=0.33416875 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.157→0.334 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 18:15 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 18:35 UTC — RL status tick
- containers: wp_judge=`Up 29 hours` | wp_consistency=`Up 29 hours`
- metrics: step=217 reward_mean=0.236284375 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.337→0.236 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 18:35 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 18:55 UTC — RL status tick
- containers: wp_judge=`Up 30 hours` | wp_consistency=`Up 29 hours`
- metrics: step=219 reward_mean=0.289365625 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.334→0.289 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 18:55 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 19:15 UTC — RL status tick
- containers: wp_judge=`Up 30 hours` | wp_consistency=`Up 29 hours`
- metrics: step=222 reward_mean=0.235471875 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.236→0.235 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 19:15 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### D · 2026-06-23 19:35 UTC — RL status tick
- containers: wp_judge=`Up 30 hours` | wp_consistency=`Up 30 hours`
- metrics: step=224 reward_mean=0.24155625 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.289→0.242 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 19:35 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### F-STEP200 · 2026-06-24 22:02 — ⚑ DECISIVE WINDOW FLAG: reward not learning (run at step 225)
- Window means: [0-50]=0.266 [51-100]=0.278 [101-150]=0.274 [151-200]=0.247. The decisive 150-200 window
  came in BELOW the ~0.275 plateau — NO upward trend over 200 steps, slight drift down (within noise band
  0.11-0.48, so not a hard collapse, but no learning signal).
- Per the user's decision rule, this FLAGS the LR/GSPO watch-item (handoff: "if flat over 100+ steps revisit
  LR / GSPO / Mechanism-3 blend"). NOT changing config mid-run.
- Guards all healthy: kl_v1=0 (on-policy), e_frac 0.952 (flat, safe), 0 errors, checkpoints 50/100/150 wrote.
  AnonPages 10045MB (just over 9900 baseline — mild, watching). Judge OK, $0 consistency.
- COST NOTE: Tinker RL training is PAID compute. ~275 steps (~1.3 days) remain. Flat reward → RL may not beat
  the v1.2 baseline at Phase-10 RLEV-01. Surfacing continue-vs-stop decision to user.

### D · 2026-06-23 19:55 UTC — RL status tick
- containers: wp_judge=`Up 31 hours` | wp_consistency=`Up 30 hours`
- metrics: step=227 reward_mean=0.25846875 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.235→0.258 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 19:55 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### DECISION · 2026-06-24 22:05 — user: CONTINUE TO 500
- Reward-flat flag acknowledged; user elects to run to 500 as instructed. Phase-10 RLEV-01 (RL vs v1.2
  baseline) is the real verdict. Checkpoints 50/100/150 saved for later eval if needed.
- Monitoring continues: guards (kl_v1, e_frac<0.5, errors, mem<2GB) + reward-not-collapsing + judge quality
  + remaining checkpoints (200/250/...). Flatness is now expected, not re-litigated each tick.

### D · 2026-06-23 20:15 UTC — RL status tick
- containers: wp_judge=`Up 31 hours` | wp_consistency=`Up 30 hours`
- metrics: step=229 reward_mean=0.239665625 min=0.0 max=1.0 non_uniform=True kl_v1=0.0 halt=None trend 0.242→0.240 over 6 rows
- warm_start: 2026-06-23 00:29:50,190 INFO __main__: WARM START from tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state: base_model=Qwen/Qwen3-30B-A3B rank=
- recent_error: none

### E · 2026-06-23 20:15 UTC — Judge-quality spot-check (local consistency endpoint)
- good=1.0 wrong=0.0 good>=wrong=True
- verdict: OK — local judge discriminates good vs wrong critique

### G · 2026-06-24 22:10 — step-250 STOP-DECISION armed + logging-reqs doc written
- New goal: at step 250, assess learning. If reward still FLAT or FALLING → save checkpoint + STOP run for
  diagnosis. (run at step 230; window[201-230]=0.266 vs [151-200]=0.247 — still ~0.26 plateau, no clear rise.)
- Decision rule at step>=250: compare window[201-250] vs [0-50]=0.266 baseline + prior plateau. If
  window[201-250] not clearly above plateau (rise > ~0.02) → FLAT → verify step-250 ckpt in manifest →
  SIGTERM trainer (clean stop preserves ckpt). Leave vLLM servers up for diagnosis/re-run.
- Future-run telemetry requirements written to 09-RL-LOGGING-REQS.md (component means, frac_groups_all_zero,
  entropy, score-dist histograms, structured tick, codified kill rule).

## Section H — Hypothesis update (real logged evidence, 2026-06-24 22:20)

**Sampling params (from launch/log):** temperature=1.0 (default, NOT lowered), group_size=4 (>1).
→ Exploration is present. Gemini's "low temperature/exploration" hypothesis is REFUTED by the params.

**fix_correctness is effectively BINARY (the key finding).** Parsing all Panickssery divergent-rollout
lines (n=15, biased toward |fix-cons|>0.3 but indicative): fix_correctness frac<0.1=0.53, frac>0.9=0.47,
**frac_mid=0.00** — a perfect step function, no intermediate values. judge_consistency is graded
(frac_mid=0.73) but only 0.3 weight.

**Unified root-cause read:** the two leading hypotheses are the SAME mechanism. The dominant 0.7
fix_correctness term is binary (solved/unsolved) → the 4 samples of a "fixable" prompt all score ~1 and an
"unfixable" prompt all score ~0 → many GSPO groups are ~uniform → normalized advantage ~0 → vanishing
gradient, even though the STEP looks non-uniform across the mixed batch. The flat ~0.27 mean ≈
(frac_fixable·0.7 + consistency·0.3), set by the prompt mix, not policy improvement. This is a REWARD-SHAPE
problem (binary dominant term), not a systems bug and not under-exploration.

**Caveat:** frac_groups_all_zero/all_one is NOT logged this run (the gap 09-RL-LOGGING-REQS.md fixes) —
can't quantify per-group collapse directly; the binary fix term is the strong circumstantial mechanism.

## Section I — Post-stop RLEV-01 protocol (armed for after the step-250 stop)

1. BEFORE stopping: confirm the latest checkpoint (step-250) is in checkpoint_manifest.json.
2. STOP trainer cleanly (kill -TERM rl_run.pid; not -9). Servers can stay or stop.
3. RLEV-01 fixed-set eval on: warmstart, step-50, step-100, step-150, step-200, step-250 (whatever saved).
   PRIMARY discriminator = judge-Spearman on data/reasoning_dataset/openai_val.jsonl (eval_judge,
   gt_mode=calibrated_canonical) per checkpoint vs the warmstart/v1.2 baseline.
   CONSTRAINT: DGX cannot host a 3rd 30B vLLM alongside judge:8000 + consistency:8001 — must STOP both
   servers first to free unified memory for the eval vLLM (:8020), OR sample each checkpoint via the Tinker
   sampling_client on the fixed val set then score offline (avoids 6 merges). Checkpoints are Tinker LoRA
   sampler_weights → merge_tinker_v3 MoE-only to serve, OR Tinker-sample-then-offline-judge (lighter).
4. DECISION: any marginal Spearman improvement across checkpoints → recipe HAS signal → targeted rerun WITH
   the 09-RL-LOGGING-REQS diagnostics. Flat Spearman → REWARD REDESIGN before more RL compute (fix the
   binary 0.7 fix_correctness term per the evidence above — e.g. graded partial-credit, rebalance weights,
   or per-group diversity shaping).
