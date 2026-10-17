# EVAL v1.1 WS-D (T5 worker-count stability)

Evaluator: independent. Worktree `agent-a66ea2785bfc14da7`, HEAD b3f6e5a, diff `5e5ea1a..HEAD`. `wpjudge.__file__` is inside the worktree (`harness/wpjudge/__init__.py`). openai 3.28.0. I edited nothing except this report. The throwaway experiment scripts live in the session scratchpad, `.../scratchpad/evd/` (`run.py`, `drive.sh`, `out/*.json`, `out/llama_{A,B,C}.log`).

## Spec coverage
| Plan item | Status |
|---|---|
| `config.workers` 4 → 1; `judge_units` default `workers=1` | done |
| entrypoint and compose `LLAMA_PARALLEL` default 1 (server, and `WPJUDGE_WORKERS`); allinone reuses the entrypoint | done; grep found no other default in code. The README still says 4, but that is WS-F's job. |
| `e2e.sh` `HOST_PARALLEL` default 1 | done |
| `judge.server_slots` reads `/props` `total_slots`, else `None` | done (stdlib, 5 s timeout, rejects bool and non-int, handles `/v1/`) |
| report `model.workers`, `server_slots`, `reproducible`; `schema_version` stays 1.0 | done |
| CLI stderr warning; ETA 5.5 units/min | done |
| unit tests (defaults, truth table, probe, jitter client) | done, and they are meaningful |
| e2e golden exact equality (dir/zip/tgz/w4), slots == 1, reproducible | done |
| `sec_determinism` (w4 → reproducible false + warning; w1 == golden) | done, **but the run order was swapped to w1 then w4** (see finding 1) |

## Findings

### 1. MAJOR: on a multi-slot server, `reproducible: true` is wrong after any concurrent run. The cause is the prompt cache, and `sec_determinism` was reordered so it cannot catch this.
**Hypothesis confirmed live, 2/2 repeats.** A w1 run that follows a w4 run on the same `--parallel 4` server differs from golden. The w1 request's prompt is an exact match for a slot whose prompt KV was computed in a batched step. The server log shows `selected slot by LCP similarity, sim_best = 1.000` with `cache_prompt` on, so llama-server reuses that KV. The batch-variant numerics then carry into a serial decode. With `cache_prompt: false` on the request, every w1 run is byte-identical to golden, including after concurrent traffic that had the cache on.

Evidence: same fixture, golden = this run's e2e `r_dir` (`-np 1`, w1: per-unit `[36,46,46,14,46,80,80,80,84,84]`, uncapped 68.8, overall 60). "raw ==" counts units whose raw model text is byte-equal to a fresh `-np 1` w1 run (C1).

| Server | Run (in order) | Units ≠ golden | Uncapped | raw == C1 |
|---|---|---|---|---|
| A: `-np 4`, cache on | A1 w4 | 8 | 67.8 | 0/10 |
| | **A2 w1** | **2** (`ser_handle_cancel_registration` 80→78, `settings-page.php:<template>` 84→82) | **68.0** | 7/10 |
| | A3 w4 | 6 | 69.7 | 0/10 |
| | **A4 w1** | **1** (`<template>` 84→82; the same unit the builder saw) | **68.3** | 8/10 |
| B: `-np 4`, fresh | B1 w4 cpf | 9 | 68.0 | 0/10 |
| | **B2 w1 cpf** | **0** | 68.8 | **10/10** |
| | B3 w4 cpf | 5 | 70.5 | 0/10 |
| | **B4 w1 cpf** | **0** | 68.8 | **10/10** |
| | B5 w4 (cache **on**) | 6 | 69.1 | 0/10 |
| | **B6 w1 cpf** | **0** | 68.8 | **10/10** |
| | B7 w4 (cache **on**) | 7 | 69.9 | 0/10 |
| | **B8 w1 cpf** | **0** | 68.8 | **10/10** |
| C: `-np 1`, fresh | C1 w1 / C2 w1 cpf / C3 w1 cpf / C4 w1 | 0 / 0 / 0 / 0 | 68.8 ×4 | 10/10 ×4 |

`cpf` = request `cache_prompt: false`. In every row, summary and dimensions equal golden exactly when the unit diff is 0. In A2 and A4 the capped overall (60) and `cap_reasons` were unchanged; only scores and the uncapped value moved. `cache_prompt:false` does not make w4 deterministic (B1 and B3 are still noisy), which matches R5.

Why this is MAJOR and not the plan's accepted "shared server" limit: no other client is involved. The failure path is the one the new warning recommends. A user runs `--workers 4` on a `--parallel 4` server and gets the WARNING, which says "Use --workers 1 ...". They rerun with `--workers 1` on the same server and get a report stamped `reproducible: true` whose scores differ from the canonical serial output. The builder found this (V11_WSD deviation 2) and then swapped `sec_determinism` to run w1 first, on a fresh server. The test now passes because of that order and does not check the property the plan wrote it for. In the plan's order (w4 then w1), the test fails.

**Fix (one line plus one test line):**
- In `judge._judge_create`, send `extra_body={"chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}`, and widen the fallback match to also catch `"cache_prompt"` (see finding 2).
- Restore the plan's order in `sec_determinism`: w4 first, then w1.

Cost: none measured (next section). Note that this is WS-C territory, so the orchestrator needs to decide who owns it.

### 2. Recommendation on always sending `cache_prompt: false`: yes
- **Throughput.** No measurable cost, because each unit prompt is distinct and only the short template prefix is shared. On a fresh `-np 1` server: cache on 109.5 s and 108.8 s, cpf 109.3 s and 109.1 s. On `-np 4` w1: cache on 110.0 s and 106.6 s, cpf 109.3 s to 110.0 s.
- **Canonical output.** Unchanged. On `-np 1`, cpf runs are byte-identical to cache-on runs (raw 10/10, C1 to C4). The golden does not move.
- **How the openai client sends it.** `extra_body` goes to `options.extra_json`, and `_base_client._build_request` merges it into the top level of the JSON body with `_merge_mappings`. So the server receives a plain top-level `"cache_prompt": false`, like the existing `chat_template_kwargs`.
- **Non-llama.cpp servers.** vLLM and Ollama ignore unknown body fields (vLLM logs "fields were present in the request but ignored"). Strict servers, such as api.openai.com, return 400 "Unrecognized request argument". wpjudge already sends the llama.cpp-only `chat_template_kwargs`. Its fallback retries once without `extra_body` when the error text contains "template". A strict server would list both arguments in the error, so the fallback still fires, and the change is no worse than today. A server that rejected only `cache_prompt` would not match "template", though, and every unit would end up unjudged. So the fallback should also match `"cache_prompt"`, or simply retry once without `extra_body` on any 400. The judge model is a GGUF served by llama.cpp in every supported deployment, so this matters only for robustness.
- **Server-side alternative.** llama-server also has `--no-cache-prompt` (`common/arg.cpp`). It could go in the entrypoint, but it only protects servers we start ourselves. The client-side field also covers a user's own multi-slot server, which is exactly the finding 1 case. The client side is the better choice.

### 3. MINOR: `reproducible` is conservative when the probe fails. Acceptable.
`/props` behind `--api-key`, or a non-llama server, gives `server_slots: null`, so `reproducible = workers == 1`. That falls back correctly. A w>1 run with ≤ 1 unit is flagged not reproducible even though it is. This is harmless. No change needed.

### 4. MINOR (docs, belongs to WS-F): the README still documents the old defaults
`README.md:36` ("Keep `--workers` equal to `--parallel`"), and lines 51, 85, 130 and 150 (defaults of 4). These must also document the finding 1 caveat if it is not fixed in code. WS-F owns this. Listed so it is not lost.

### 5. Note: the fixture `build/` gitignore (builder deviation 3)
This is already fixed on main by f0cccf0 ("track fixture build dir"). The file was present in this worktree. Nothing to do for WS-D.

### Code review notes (no defects)
- `server_slots` uses `re.sub(r"/v1/?$", "", base_url.rstrip("/"))`, which is correct for `/v1` and `/v1/`. `json.load(r).get` on a non-dict raises, and the broad except catches it.
- `aggregate` falls back to `cfg.workers` when `model_meta` has no workers, which keeps old callers and tests working. The `test_aggregate.py` superset change is minimal and justified.
- `same()` in chk.py compares per-unit (score, scores, verdict), summary (overall, uncapped, cap_reasons) and dimensions, and it lists the differing ids. It is a good strict check.
- The diff is small and matches the surrounding style.

## Commands run
- `.venv/bin/python -m pytest tests/ -q` → **49 passed**.
- `GGUF=…/shrink-iq4xs-im_f16.gguf WORK=<scratch>/evd/work flock /tmp/wpjudge-gpu.lock tests/e2e.sh cli determinism` → **98 passed, 0 failed**.
  - Golden: dir == zip == tgz == w4 exactly, and every report has `server_slots 1`, `reproducible true`.
  - Determinism: w4 on `-np 4` reported `server_slots 4`, `workers 4`, `reproducible false`, and printed the warning. Text: `wpjudge: WARNING: workers=4 on a server with 4 slots decodes concurrently; ...`. ETA line: `10 units, ETA ~1.8 min`.
  - w1 (run first, on the fresh `-np 4` server) == golden exactly.
- `flock /tmp/wpjudge-gpu.lock <scratch>/evd/drive.sh` ran three servers on port 8199 (A: `-np 4`, B: `-np 4`, C: `-np 1`, each with `-c` = np × 8192) and produced the table in finding 1. The patched `_judge_create` was only in the throwaway script, and nothing was committed. Afterwards `nvidia-smi` showed no compute apps. Every server I started was killed.

## Verdict rationale
The implementation matches the plan and is clean. The live e2e passes. However, `reproducible: true` is demonstrably false in the workflow the warning itself recommends, and the e2e was reordered so it hides this (finding 1). The fix is one proven line, it costs no throughput, and the e2e order needs restoring.

VERDICT: FAIL
