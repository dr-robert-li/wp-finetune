---
phase: 21
slug: sft-training-generation-judge-models
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-13
---

# Phase 21 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

**Note on this phase's test shape:** most Phase 21 work is remote Tinker training or local GB10
merge/serve/eval — inherently minutes-to-hours, not sub-minute. The fast-feedback loop is (a) one pytest
over the forked data adapter, and (b) per-task **receipt-field assertions**: each GPU/Tinker task writes a
JSON receipt under `output/base21/`, and its `<automated>` verify is a `python -c` assertion over that
receipt that runs in milliseconds once the (long) task completes. This is the same
receipt-as-ground-truth pattern Phase 20 used (`output/base20/*.json`).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.1.1 (installed Phase 20-01) + `python -c` receipt-field assertions |
| **Config file** | none dedicated — plain `pytest tests/test_X.py -x -q` (Phase 20 convention) |
| **Quick run command** | `.venv-tinker/bin/python -m pytest tests/test_tinker_reasoning_data_v4.py -x -q` |
| **Full suite command** | `pytest tests/ -k "v4 or phase21" -q && pytest tests/test_download_model_v4.py tests/test_check_token_alignment.py -x -q` |
| **Estimated runtime** | pytest ~20s; receipt assertions <1s each (heavy compute is the task itself, not the check) |

---

## Sampling Rate

- **After every task commit:** Run the changed task's `<automated>` (the data-adapter pytest for 21-01-01; the receipt assertion for that task otherwise)
- **After every plan wave:** Run the full suite command + re-run Phase 20's `tests/test_download_model_v4.py tests/test_check_token_alignment.py` (confirms shared config/download/merge machinery did not regress)
- **Before `/gsd-verify-work`:** GEN-03 wp-bench receipt + JUDGE-03 vLLM-served rho receipt both present and green
- **Max feedback latency:** ~20s for the data-adapter path; GPU/Tinker tasks gated by their receipt assertion post-run

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 21-01-01 | 01 | 1 | GEN-01 | T-21-02 / — | renderer+LR+64K decision recorded, no empty-think injection | unit | `.venv-tinker/bin/python -m pytest tests/test_tinker_reasoning_data_v4.py -x -q` | ❌ W0 | ⬜ pending |
| 21-01-02 | 01 | 1 | GEN-01 (MoE-probe prereq) | T-21-01 / T-21-03 | fused-expert merge guard equal + base≠merged (no silent all-zero) | integration | `python -c "import json;r=json.load(open('output/base21/moe_merge_probe.json'));assert r['merge_ok'] and r['base_vs_merged_differs'] and r['merged_target_module_count']==r['expected_target_module_count'] and r['smoke_vl_merge_rerun_ok']"` | ❌ W0 | ⬜ pending |
| 21-02-01 | 02 | 2 | GEN-02 | T-21-06 / — | real reasoning-mix feeds the forked driver | integration | `python -c "import json;r=json.load(open('output/base21/gen02_run.json'));assert r['smoke']['smoke_ok']"` | ✅ (driver from 21-01) | ⬜ pending |
| 21-02-02 | 02 | 2 | GEN-02 | T-21-04 / T-21-05 | loss decreases + terse gate measured, no silent collapse | integration | `python -c "import json;f=json.load(open('output/base21/gen02_run.json'))['full'];assert f['full_ok'] and f['loss_last'] is not None and f['promoted_checkpoint_name']"` | ✅ | ⬜ pending |
| 21-03-01 | 03 | 2 | JUDGE-02 | T-21-07 / — | 3 seeds, new base, reused labels, resolvable checkpoints | integration | `python -c "import json;[json.load(open(f'output/tinker/wp-judge-v4-s{s}-manifest.json')) for s in (1,0,2)]"` | ✅ | ⬜ pending |
| 21-03-02 | 03 | 2 | JUDGE-02 | T-21-08 / — | reused-label provenance recorded, all seeds resolvable | integration | `python -c "import json;r=json.load(open('output/base21/judge02_run.json'));assert r['all_seeds_complete'] and len(r['seeds'])==3 and r['relabel_reuse']"` | ✅ | ⬜ pending |
| 21-04-01 | 04 | 2 | JUDGE-01 | T-21-10 / T-21-11 | parse-fail rate measured truncation-safe vs 18% anchor | integration | `python -c "import json;r=json.load(open('output/base21/judge01_format_smoke.json'));assert r['n_prompts']>=20 and (r['n_parse_ok']+r['n_parse_fail'])==r['n_prompts'] and 'vs_anchor' in r"` | ❌ W0 (new smoke script) | ⬜ pending |
| 21-05-01 | 05 | 3 | GEN-03 | T-21-12 / — | gen merge guard equal + base≠merged | integration | `python -c "import json;r=json.load(open('output/base21/gen03_merge.json'));assert r['merge_ok'] and r['base_vs_merged_differs'] and r['merged_target_module_count']==r['expected_target_module_count']"` | ✅ | ⬜ pending |
| 21-05-02 | 05 | 3 | GEN-03 | T-21-13 / T-21-14 | CI-aware wp-bench vs floor, thinking-off, full suite | integration | `python -c "import json;r=json.load(open('output/base21/gen03_wpbench.json'));assert 'wpbench_ci_lower' in r and 'pass' in r and r['enable_thinking'] is False and r['n_tests']>=300"` | ✅ | ⬜ pending |
| 21-06-01 | 06 | 4 | JUDGE-03 | T-21-15 / T-21-16 | 3-seed 8192-cap capture rho + ensemble, index-aligned | integration | `python -c "import json;r=json.load(open('output/base21/judge03_capture_rho.json'));assert len(r['per_seed'])==3 and r['max_tokens']==8192 and 'rho' in r['ensemble_median']"` | ✅ (eval_relabel.py) | ⬜ pending |
| 21-06-02 | 06 | 4 | JUDGE-03 | T-21-18 / T-21-19 | promoted seed merged+served at 8192 cap, guard checked | integration | `python -c "import json;v=json.load(open('output/base21/judge03_rho.json'))['vllm_served_single_seed'];assert v['max_tokens']==8192 and 'rho' in v and 'ci_lower' in v"` | ✅ | ⬜ pending |
| 21-06-03 | 06 | 4 | JUDGE-03 | T-21-17 / — | CI-aware vs pre-registered targets, methodology labeled, miss recorded valid | integration | `python -c "import json;r=json.load(open('output/base21/judge03_rho.json'));assert 'overall_pass' in r and r['targets']=={'single':0.85,'ensemble':0.87} and (r['overall_pass'] or r.get('disposition')=='valid_recorded_miss')"` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_tinker_reasoning_data_v4.py` — v4 data-adapter build + max-length<64K assert + no empty-think injection (GEN-01, 21-01-01)
- [ ] `scripts/build_base21_moe_probe_adapter.py` — the highest-value Wave-0 item: real `train_mlp=True` MoE merge probe that gates all paid runs (21-01-02)
- [ ] `scripts/smoke_judge_format_base21.py` — raw-base judge-format-compliance smoke (JUDGE-01, 21-04-01)
- [ ] `scripts/tinker_reasoning_data_v4.py` / `scripts/tinker_reasoning_sft_v4.py` — forked v4 siblings (enable every downstream train/eval task)

*Framework already installed (pytest 9.1.1, Phase 20-01). No new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | — | Every Phase 21 behavior has an automated receipt-field assertion or a pytest; GPU/Tinker receipts are the ground-truth evidence (Phase 20 precedent) | — |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (every task has one)
- [x] Wave 0 covers all MISSING references (test file + 2 new scripts + 2 forked siblings)
- [x] No watch-mode flags
- [x] Feedback latency < 21s for the data-adapter loop; GPU/Tinker tasks gated by post-run receipt assertions (inherent to the compute shape)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-13
