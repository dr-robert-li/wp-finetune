# Phase 5 run — tests/e2e.sh

Date: 2026-10-10, 12:45–13:29 AEST. Host: GB10 aarch64, driver 580.
Model: `shrink-iq4xs-im_f16.gguf` (IQ4_XS).
WORK: `harness/build_notes/e2e_work/`. Combined log: `e2e_work/e2e.log`. Per-section output: `e2e_work/run_<section>.out`.
No harness, test, Docker or fixture file was edited.

## Result per section

| Section | PASS | FAIL | Wall |
|---|---|---|---|
| malicious | 17 | 0 | 4 s |
| serverdown | 18 | 0 | 1 s |
| cli | 87 | 0 | 573 s |
| library | 33 | 0 | 257 s |
| real | 28 | 0 | 570 s |
| parity (run 1) | 2 | 1 | 351 s |
| parity (re-run) | 2 | 1 | 352 s |
| api | 41 | 0 | 114 s |
| allinone | 17 | 0 | 86 s |
| **Total (first runs)** | **243** | **1** | |

Total runtime: 44 min wall clock. That is about 39 min of suite runs (parity counted twice) plus about 3 min for one diagnostic parity pass.
GPU hygiene: before every section and after every section, no llama-server process, no GPU compute app, and no container (wpjudge-e2e or other) was present. The `post:` line at the end of each `run_<section>.out` records this.

## FAIL lines (verbatim) and diagnosis

```
FAIL parity: overall |d| max 8 <= 6 (mean 1.50)
```
(the same line in both runs; `PASS parity: verdicts equal 16/16` in both runs)

**Diagnosis: environment (model-server build drift). Not a harness bug. Not a test-script bug.**

- Evidence of determinism: the re-run gave byte-identical `parity_host.json` and `parity_container.json` (`cmp` equal to the `*.run1.json` copies). So the failure is not transient.
- Per-prompt overall scores, host (build 9954 / 8f114a9b5) vs container (b11515):
  79/78, 84/82, 80/80, 74/74, 70/70, 74/72, 62/60, 80/80, 79/78, 72/68, **82/74**, 78/76, 78/78, 66/64, 80/80, 80/80.
  Only prompt 11 goes past the limit (Δ 8). The other 15 have Δ ≤ 4. The host score is ≥ the container score on all 16 prompts. That is a small systematic offset between the two llama.cpp builds.
- Slot config is ruled out. e2e.sh:61-63 starts the host with `--parallel 4 -c 32768`, while the container uses `--parallel 1 -c 8192` (e2e.sh:387-389). In a diagnostic pass I started the host build with `--parallel 1 -c 8192` on port 8099 (`e2e_work/parity_host_p1.json`, log `llama-host-p1.log`). The 15 parsed scores were identical to the host `--parallel 4` scores (max |Δ| 0). So the Δ 8 comes from the build difference (9954 vs 11515), not from slots. One side note: in that diagnostic pass, prompt 14 returned no parseable `<judge_output>` (verdict null). This did not happen in either suite run. It affects only the diagnostic pass.
- The check itself is correct. The parity check is in e2e.sh:380-395 and chk.py `cmd_parity_cmp` (e2e.sh:235-240). It uses the |Δ| ≤ 6 limit from PLAN.md binding change 13.
- Impact: the shipped path is the container (b11515). The api and allinone sections run on that build, and they pass with the same verdicts and cap behaviour as the host. Options for the orchestrator: (a) accept this as a known 1/16 drift, (b) rebuild the host llama.cpp at b11515 and re-run parity, or (c) change the limit in the plan. I cannot choose: the limit is a plan decision.

## Recorded numbers

### Fixture (vuln-plugin, 10 units, 5 skips, 7 PHP files)
| Report | overall | uncapped | capped | cap_reasons | verdicts |
|---|---|---|---|---|---|
| cli dir / zip / tgz (workers=1) | 60 | 68.8 | true | 8 | 5 PASS / 5 FAIL |
| cli dir workers=4 | 60 | 69.1 | true | 8 | 5 / 5 |
| library dir / zip | 60 | 68.8 | true | 8 | 5 / 5 |
| api (container b11515) | 60 | 67.7 | true | 8 | 5 / 5 |
| allinone (container b11515) | 60 | 69.7 | true | 8 | 5 / 5 |

- dir vs zip vs tgz: max |Δ overall| = 0. workers=4 vs workers=1: verdicts equal.
- Planted SQLi caps in all 3 golden reports (`ser_search_registrations`, `ser_delete_registration`; the pre-run capped 2).
- Every planted unit scored below every clean unit: planted [36, 46, 46, 14, 46], clean [80, 80, 80, 84].

### Real targets
| Target | Units (plan) | Skipped | Coverage | Wall | ETA printed | Overall | Capped | Verdicts |
|---|---|---|---|---|---|---|---|---|
| classic-editor (plugin) | 39 (39) | 1 | 100.0% | 221 s | ~3.5 min | 74.9 | false | 39 PASS |
| twentytwentyfour (theme) | 60 (60) | 0 | 100.0% | 337 s | ~5.5 min | 81.7 | false | 60 PASS |

- classic-editor: all 39 units are kind=method, mean 75.3 (LOC-weighted 74.9, 892 LOC).
- **Theme split (binding change 15):** kind=template n=57, mean overall 81.6 (LOC-weighted 81.6, 3329 LOC). kind=function n=3, mean 84.0 (LOC-weighted 84.0, 152 LOC). The theme has no kind=method units. Templates score 2.4 points below functions.
- Both reports pass the independent recompute of all summary numbers from `functions[]` in chk.py.

### Parity (binding change 13)
- Driver-580 smoke: PASS (`version: 0.6.0-dev (build 11515, commit 3d65c90d0)`).
- Verdicts: 16/16 equal (all PASS).
- Overall: max |Δ| = 8 (limit 6), mean |Δ| = 1.50. Result: **FAIL**. See the diagnosis above.

### API (binding change 12) — all pass
Partial `/functions` returned results while the job ran. The webhook was delivered. Uploads were rejected with the expected codes: .py 415, 60 MB 413, file:// webhook 400, non-numeric Content-Length 400, `..%2f` paths 404. A zip-slip job failed cleanly. Auth on/off worked, including on `/functions`. A done job survived a restart. A running job became `failed: interrupted by server restart; resubmit`. Retention swept the 8-day-old job and kept the 6-day-old job.

## Report paths (in `harness/build_notes/e2e_work/`)
- CLI: `r_dir.json`, `r_zip.json`, `r_tgz.json`, `r_w4.json`, `r_stdout.json` (stderr in `r_*.err`)
- serverdown: `r_down.json`
- library: `lib_dir.json`, `lib_zip.json`
- real: `real/classic-editor.json`, `real/twentytwentyfour.json` (plus `.err` and the downloaded `.zip` files)
- parity: `parity_host.json`, `parity_container.json` (`*.run1.json` = first run), `parity_host_p1.json` (diagnostic)
- api: `api_report.json`, `hook.json`, `j2.json`
- allinone: `aio_report.json`
- malicious: `mal/` (generated archives)
- host server logs: `llama-host.log`, `llama-host-p1.log`
- Note: `big.zip` (60 MB random data, from the api 413 test) is still in WORK. Delete it if it should not be kept.

## Resolution (orchestrator, 2026-10-10)
- Evaluator verdict: PASS WITH NOTES (`EVAL_PHASE5.md`). All numbers were independently recomputed with zero discrepancies.
- Parity: the user widened binding change 13 to |Δ| ≤ 8 (cross-build drift, not slot noise). `e2e.sh:240` was updated. Because the outputs are byte-deterministic, `chk.py parity-cmp` was re-run on the saved files: `PASS parity: overall |d| max 8 <= 8 (mean 1.50)`.
- Phase 6 docs must cover:
  - the cli, library and real numbers come from host build 9954, and the shipped container is b11515;
  - the restart and retention behaviour;
  - the template caveat (templates score 81.6 vs 84.0 for functions; `WPJUDGE_TEMPLATES=0`);
  - runaway generations to max_tokens can happen and become unjudged after the retry.
