---
slug: gb10-device-map-auto-oom
status: resolved
trigger: "profile_v4_judge.py (Phase 25-01) killed at ~62% model load; make the fix persist across all prior run sites so this failure mode does not recur"
created: 2026-07-15
updated: 2026-07-15
---

# Debug: GB10 unified-memory `device_map="auto"` OOM

## Symptoms

- `profile_v4_judge.py` (PID 2474645) died silently at ~62% of a 67 GiB model
  load during Phase 25-01. Log `logs/sieve-v4/profile.log` ends mid-`Loading
  weights` with no traceback; 0/5 required artifacts written.
- Loader guard (T-25-01) PASSED first (0 missing keys) — the checkpoint is sound;
  only the subsequent full materialization died.

## Root Cause (CONFIRMED — kernel evidence)

Global OOM-kill at 16:56:23:
```
Out of memory: Killed process 2474645 (python) total-vm:153434072kB,
anon-rss:57036580kB ... oom_score_adj:100  (constraint=CONSTRAINT_NONE, global_oom)
```
On this GB10, `torch.cuda.mem_get_info()` reports **total=130.7 GB, free=118.3 GB**
— the entire 121 GiB *unified* pool as "GPU-free". `device_map="auto"` therefore
treats "cuda:0 ~118 GiB" and "cpu ~110 GiB" as two independent pools (~230 GiB)
and balances the model across BOTH, placing part on `cpu`. But CPU and GPU are
the SAME physical RAM here, so the ~54 GiB CPU-resident shards + the unified-GPU
shards collided → global OOM. GPU-mapped pages are pinned/unswappable, so the
16 GiB swap could not rescue it. `oom_score_adj=100` (from the waveterm snap
scope the process inherited) made it the preferred victim.

Not caused by competing processes (the rest of the process table was a fleet of
sub-135 MB MCP/python procs; 111 GiB was free once it died). Same trap that
OOM'd the Phase 4.4 P0-v2 Unsloth load.

## Fix (applied — root-cause, all sites)

New shared helper `scripts/sieve_arch.gb10_load_kwargs()` returns
`{"device_map": {"": 0}, "low_cpu_mem_usage": True}` (CUDA-absent → `{"": "cpu"}`):
single-device placement removes the phantom CPU+GPU split; `low_cpu_mem_usage`
streams shards via meta-init instead of a full CPU materialization. A 67 GiB
bf16 model then occupies the unified pool once (~67/121 GiB), leaving room for
bs=1 activations.

Every `device_map="auto"` full-model load routed through the helper (6 sites):
- scripts/profile_v4_judge.py       (the OOM site)
- scripts/profile_merged_model.py   (standalone loader for the same profile)
- scripts/profile_base_model.py
- scripts/sieve_v4_tooling_smoke.py
- scripts/run_eval_triage.py
- scripts/prepare_tokenizer.py       (×2 sites)

Intentional `device_map="cpu"` merge/smoke scripts (merge_adapter, merge_tinker_v3,
smoke_load_base20, check_token_alignment, build_base20_probe_adapter) left as-is.

## Durability guard

`tests/test_gb10_load_safety.py` — AST-scans every `scripts/*.py` and fails if any
`from_pretrained` reintroduces a bare `device_map="auto"` (comment/docstring-proof),
plus asserts the helper's contract. Prevents silent regression in future edits.

## Verification

- `python -m scripts.sieve_arch --self-check` → OK
- `pytest tests/test_gb10_load_safety.py` → 150 passed (all scripts clean + contract)
- `pytest tests/test_sieve_*.py` → 28 passed (no import regression)
- In-situ relaunch: profiling load passes the former ~62% death point (see resolution log).

files_changed:
  - scripts/sieve_arch.py
  - scripts/profile_v4_judge.py
  - scripts/profile_merged_model.py
  - scripts/profile_base_model.py
  - scripts/sieve_v4_tooling_smoke.py
  - scripts/run_eval_triage.py
  - scripts/prepare_tokenizer.py
  - tests/test_gb10_load_safety.py
