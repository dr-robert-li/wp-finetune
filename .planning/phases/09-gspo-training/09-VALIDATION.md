---
phase: 9
slug: gspo-training
status: planned
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-20
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing; established Phase 8) |
| **Config file** | pyproject.toml / pytest.ini (existing) |
| **Quick run command** | `pytest tests/test_rl_train.py tests/test_rl_rollouts.py tests/test_rl_judge_dispatch.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~30s (RL-loop unit tests are synthetic-tensor only; NO live Tinker) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_rl_train.py tests/test_rl_rollouts.py tests/test_rl_judge_dispatch.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | GRPO-05 | T-09-POISON / T-09-LEAK | Prompts only from audited 4.2 corpus; no val leakage (PROVENANCE.md) | integration | `python scripts/build_rl_prompts.py && test -s data/rl_prompts/wp_gen_train.jsonl` | ❌ W0 | ⬜ pending |
| 09-01-02 | 01 | 1 | GRPO-05 | — | Prompt-only adapter, importable without tinker | unit | `python -c "import scripts.tinker_rl_data as d; assert d.RENDERER_NAME=='qwen3_disable_thinking'"` | ❌ W0 | ⬜ pending |
| 09-02-01 | 02 | 1 | GRPO-06/07/08 | — | MoE-metric mock fixture (forward_backward + forward_backward_custom) for offline loop tests | unit | `python -c "import ast; ast.parse(open('tests/conftest.py').read())" && grep -q mock_tinker_client tests/conftest.py && grep -q forward_backward_custom tests/conftest.py` | ✅ (conftest exists) | ⬜ pending |
| 09-02-02 | 02 | 1 | GRPO-05/06/07/08 | — | 8 named contract stubs collect green offline; test_gspo_rspo_floor asserts GSPO is the default loss path | unit | `pytest tests/test_rl_train.py --collect-only -q` | ❌ W0 | ⬜ pending |
| 09-02-03 | 02 | 1 | GRPO-06/07/08 | T-09-STALE | Zero dgx.execute in Phase 9 ROADMAP block; deviations surfaced; GSPO-primary/GRPO-fallback stated | grep-gate | `awk '/^### Phase 9:/{f=1}/^### Phase 10:/{f=0}f' .planning/ROADMAP.md \| grep -v '^#' \| grep -ci 'dgx.execute("unsloth_studio"'` == 0 | ✅ (ROADMAP exists) | ⬜ pending |
| 09-03-01 | 03 | 2 | GRPO-05 | T-09-RWD-HACK / T-09-INJECT / T-09-SELFPREF | Subprocess (not Anthropic API / not Agent); rubric; cache | unit | `pytest tests/test_rl_judge_dispatch.py -k cache -q` | ❌ W0 (09-02 stubs) | ⬜ pending |
| 09-03-02 | 03 | 2 | GRPO-05 | T-09-NOISE | Parallel gather; 120s timeout; group-mean imputation | unit | `pytest tests/test_rl_judge_dispatch.py -q` | ❌ W0 | ⬜ pending |
| 09-04-01 | 04 | 2 | GRPO-05 | T-09-RWD-CAP / T-09-SECDROP | judge>=gen; consistency cap<=0.5; security-group drop; pipeline unmodified | unit | `pytest tests/test_rl_rollouts.py -k "interleave or judge_ge_gen or cap" -q` | ❌ W0 | ⬜ pending |
| 09-04-02 | 04 | 2 | GRPO-05/07 | T-09-NOISE | Delegated cookbook advantages; mixed->nonzero, constant->dropped | unit | `pytest tests/test_rl_rollouts.py -q` | ❌ W0 | ⬜ pending |
| 09-05-01 | 05 | 3 | GRPO-06/07 | T-09-CRED | Frozen-router LoRA (no train_router); GSPO+RSPO floor PRIMARY by default (clamp min=1.0, active with NO flag — SC3); dry-run drives the default GSPO path; GRPO fallback reachable | unit | `pytest tests/test_rl_train.py -k "lora_config or rspo or gspo" -q && python scripts/rl_train.py --dry-run --total-steps 1` | ❌ W0 | ⬜ pending |
| 09-05-02 | 05 | 3 | GRPO-08/06 | T-09-ROUTE / T-09-DIVERGE / T-09-CKPT | Per-step KL+MoE auto-halt; monitor-only Jaccard (corrected mask path); persistent ckpt | unit | `pytest tests/test_rl_train.py -k "kl_autohalt or routing_autohalt or protected_mask" -q` | ❌ W0 | ⬜ pending |
| 09-06-01 | 06 | 4 | GRPO-05/06/07/08 | T-09-RWD-REG / T-09-STALE | Tinker-native skill; zero DGX; deviations; anti-hack regression gate | grep-gate | `test "$(grep -v '^>' '.claude/skills/wp-finetune:run-rl-training/SKILL.md' \| grep -ci 'dgx\|unsloth\|docker exec')" = 0` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

> RL-loop math (GSPO sequence-IS ratio, RSPO stop-gradient floor, router-shift / KL-sample-train
> computation, protected-expert retention check, judge-reward noise/CV measurement) is unit-tested
> on synthetic tensors via tests/test_rl_train.py + mock_tinker_client WITHOUT a live Tinker run —
> these are the Nyquist-sampled invariants. Full Tinker training runs are manual (below).

---

## Wave 0 Requirements

- [ ] `tests/test_rl_train.py` — 8 named stubs covering GRPO-05/06/07/08 (09-02 Task 2); test_gspo_rspo_floor asserts GSPO default path
- [ ] `tests/conftest.py` — add `mock_tinker_client` fixture mocking forward_backward AND forward_backward_custom (09-02 Task 1; file exists)
- [ ] `tests/test_rl_judge_dispatch.py` — cache + timeout/impute tests (09-03; owned by 09-03)
- [ ] `tests/test_rl_rollouts.py` — interleave/cap/advantage tests (09-04; owned by 09-04)
- [ ] `data/rl_prompts/wp_gen_train.jsonl` + `wp_judge_train.jsonl` (09-01)
- [ ] ROADMAP Phase 9 skill text DGX→Tinker correction + GSPO-primary/GRPO-fallback statement (09-02 Task 3)

*pytest framework already installed (Phase 8) — no install task.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end RL training run on Tinker | GRPO-05/06 | Cloud GPU cost + wall-clock; cannot run in CI | `python scripts/rl_train.py --dry-run` first (exits 0, drives the default GSPO path), then real run via the run-rl-training skill; inspect output/rl_checkpoints/metrics/rl_metrics.jsonl per-step |
| Auto-halt on KL / MoE-routing breach (live) | GRPO-07/08 | Requires live rollout/train divergence | Verify halt fires from the per-step monitor on a real threshold breach and writes an emergency checkpoint (synthetic check_halt path is unit-tested) |
| Protected-expert Jaccard vs Phase 7 mask (live routing) | GRPO-06 | Needs real routing distributions from a Tinker run | Every-N-step Jaccard logged to rl_metrics.jsonl; review vs Phase 7 baseline (monitor-only) |
| Panickssery self-preference spot-check (R1) | GRPO-05 | Human judgment on divergence batches | Every ~50 steps, human-review 5 of the >0.3-divergence rollouts per D-09-05 R1 |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
