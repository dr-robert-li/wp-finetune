# V11 WS-D (T5 worker-count stability): builder notes

Branch `worktree-agent-a66ea2785bfc14da7`, worktree `/home/rli/Desktop/projects/wp-finetune/.claude/worktrees/agent-a66ea2785bfc14da7`. Base: main 5e5ea1a (plan + research + brief).

## What changed
- `wpjudge/config.py` `Config.workers`: 4 → 1.
- `docker/llama-entrypoint.sh`: `P=${LLAMA_PARALLEL:-1}` (was 4).
- `docker-compose.yml`: both `${LLAMA_PARALLEL:-1}` (server env and harness `WPJUDGE_WORKERS`).
  - The allinone image uses the same entrypoint (`Dockerfile` copies it; `allinone.sh` sets no parallel or workers), so it is serial too. No other `PARALLEL`/`WORKERS` default exists (grepped Dockerfile, allinone.sh, api.py).
- `wpjudge/judge.py`:
  - new `server_slots(base_url)`: GET `<base_url minus /v1>/props`, returns `total_slots` if it is an int; `None` on any error, a missing key, a bool or non-JSON. Uses stdlib `urllib.request`, 5 s timeout.
  - `judge_units` default `workers=1`.
- `wpjudge/__init__.py` `scan`: `model_meta` adds `workers` (cfg) and `server_slots` (probe).
- `wpjudge/aggregate.py` `build_report`: `model` adds `workers` (from model_meta, falling back to `cfg.workers`), `server_slots`, and `reproducible = workers == 1 or server_slots == 1`. `schema_version` stays "1.0".
- `wpjudge/cli.py`: the ETA uses 5.5 units/min (was 11). When `model.reproducible` is false, it prints one stderr line: `wpjudge: WARNING: workers=N on a server with S slots decodes concurrently; scores and cap_reasons may change between runs. Use --workers 1 or a --parallel 1 server for a reproducible report.`
- `tests/test_config.py` (new): serial defaults (config, entrypoint, compose); the `reproducible` truth table (1,4)(4,1)(1,None)(1,1) → True and (4,4)(4,None) → False; the `/props` probe against a local stdlib HTTP server (4, 1, missing key, bool, non-JSON, `/v1/` trailing slash, server down); and a jitter fake client where `judge_units` w1 == w4 (client side only).
- `tests/e2e.sh`:
  - header: `determinism` in the section list, `HOST_PARALLEL (1)`, `DET_PORT (8099)`.
  - `HOST_PARALLEL=${HOST_PARALLEL:-1}`, `DET_PORT=${DET_PORT:-8099}`.
  - chk.py: new `same(a, b, msg)`, which requires exact equality on per-unit `overall_score`/`scores`/`verdict`, `summary` `overall_score`/`overall_score_uncapped`/`cap_reasons`, and `dimensions`, and prints the differing unit ids.
  - `cmd_golden`: the dir-vs-zip/tgz "≤ 6" check and the "w4 verdicts equal" check are replaced by `same` (dir vs zip, dir vs tgz, w1 vs w4). Every report (dir, zip, tgz, w4) must have `model.server_slots == 1` and `reproducible == true`.
  - new `cmd_determinism` and `sec_determinism`. It fails clearly when `$WORK/r_dir.json` is absent. It runs `host_down`, then starts its own `--parallel 4 -c 32768` server on `DET_PORT` (it fails if the port is already in use). It runs w1 first, then w4. Checks: w4 has `server_slots 4`, `workers 4`, `reproducible false` and the stderr WARNING; w1 has no warning, `reproducible true`, and is exactly equal (`same`) to the golden `r_dir`. The server pid goes into `HOST_PID`, so `host_down` and the EXIT trap stop it.
  - `determinism` is added to the `case` list and to `all` (after `real`, before `parity`).

## Deviations from the plan
1. **`tests/test_aggregate.py:26`, one line** (owned by B/C): `r["model"] == MODEL` → `r["model"].items() >= MODEL.items()`. Without it, the existing test breaks on the new model fields. Expect a trivial merge.
2. **`sec_determinism` runs w1 before w4** (the plan lists w4 first). Live run 1 used w4 then w1 on the same `-np 4` server, and w1 then differed from golden on one unit (`templates/settings-page.php:1:<template>`: 84 → 82, plus an extra `error_handling: 6`; uncapped 68.8 → 68.3). Run 2 used w1 on the fresh server and was exactly equal. Likely cause: w1 reused a slot whose cached prompt KV had been computed during the batched w4 run (`cache_prompt` slot reuse), so batch-variant numerics leaked into a serial request. **Finding for the orchestrator and docs (WS-F):** `reproducible=true` for w1 on a multi-slot server holds only if no concurrent traffic has touched it since startup, not just during the run. This widens R5's "shared multi-slot server" hole. The README advice "use a dedicated `--parallel 1` server" covers it. A client-side `cache_prompt:false` might close it (R5 showed `-np 4`/w1 with `cache_prompt:false` == ref), but that is a `judge._judge_create` change (WS-C territory) and was not tested here after concurrent traffic. Not built.
3. **Environment, not committed:** `tests/fixtures/vuln-plugin/build/index.asset.php` is gitignored by the repo-root `.gitignore:10 build/`, so the fixture in a fresh worktree or clone is missing it. Run 1 failed 12 skip-row/`files_scanned` checks because of this (6 files instead of 7). I copied the file from the main checkout into this worktree (untracked) for run 2. **Every other worktree, and a fresh clone, will hit the same failure.** The fix is `git add -f` of that file, or a `!harness/tests/fixtures/**/build/` negation. That belongs to WS-E or the orchestrator.
4. The plan says "do not commit"; the brief says to commit on the worktree branch. I followed the brief.

## Test results
- `.venv/bin/python -m pytest tests/` → **49 passed** (`wpjudge.__file__` is in this worktree; pytest 9.1.1).
- `bash -n tests/e2e.sh` → ok.

## Live run (evidence)
Command: `GGUF=…/models/_gguf/shrink-iq4xs-im_f16.gguf WORK=<scratchpad>/wsd_work flock /tmp/wpjudge-gpu.lock tests/e2e.sh cli determinism` (host llama-server b9952-2).
- Run 2 (final code): **98 passed, 0 failed.** Started 18:47:19. cli ran 18:47 to 18:56:31; determinism finished 18:59:32.
  - golden: dir == zip == tgz == w4 **exactly** (units differing []; summary and dimensions equal). All have `server_slots 1`, `reproducible true`. Per-unit scores `[36,46,46,14,46,80,80,80,84,84]`, uncapped 68.8, which matches R5's `ref`.
  - Serial fixture time per run: about 105 to 110 s for 10 units (about 5.5 units/min). The ETA printed `~1.8 min`.
  - determinism: w1 on `-np 4` == golden exactly (`server_slots 4`, `reproducible true`, no warning). w4 on `-np 4` gave `server_slots 4`, `reproducible false`, and the warning. Its scores were `[36,42,46,42,46,80,78,80,82,82]`, uncapped 69.2, which shows the nondeterminism the flag reports.
  - After stop: no compute app of ours left. A later pid 2755417 on another port belongs to another workstream.
- Run 1 (w4 before w1, fixture file missing): 85 passed, 13 failed. 12 were the fixture `build/` file (deviation 3). The other was the w1-after-w4 mismatch (deviation 2). Logs are kept in the scratchpad under `wsd_work_run1/`.
