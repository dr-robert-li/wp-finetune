# R5 (T5): worker-count stability

## Answer
The scores change with the worker count because **llama.cpp's CUDA kernels are not batch-invariant**. When several sequences decode in the same step, matmuls run with `ncols_dst` = tokens in the batch. `ggml-cuda/mmvq.cu` picks `nwarps` and `rows_per_block` from `ncols_dst` (`calc_nwarps` / `calc_rows_per_block`, lines 352–470), and `ne11 > 8` switches to MMQ. A different batch composition gives a different float reduction order, so a different argmax token, and the generation diverges from that token on.

No sampling setting, cache setting or server flag changes this:
- llama.cpp has no batch-invariant or deterministic option in host build b9952-2 (the "9954" tree) or container b11515. Checked `--help` and grepped the source and the container `.so` files.
- The README for the `cache_prompt` server option says logits are not guaranteed identical across batch sizes.

Decoding **one sequence at a time is fully deterministic**. It gives the same bytes on every run, across server configs, with prompt caching on or off, and across separate server processes.

**Recommendation:** serialize decoding by default, with `--parallel 1` on the server and `workers=1` in the client. The cost is 46% of throughput (10.0 → 5.4 units/min). Keep `>1` as an opt-in fast mode that the report stamps as not reproducible.

## Method
- Fixture: `tests/fixtures/vuln-plugin`, 10 units, temperature 0, `max_tokens` 2048.
- Host `llama-server` b9952-2, model `shrink-iq4xs-im_f16.gguf`, flags `-ngl 999 --jinja --chat-template-kwargs '{"enable_thinking": false}'` plus the flags under test.
- Each run calls the real `extract.collect_units`, `judge.judge_units` and `aggregate.build_report`. Only `_judge_create` is patched, to add `extra_body` options.
- Every server ran under `flock /tmp/wpjudge-gpu.lock`, with `nvidia-smi` checked before and after. All servers were killed afterwards.
- Scripts and raw outputs are in the session scratchpad `…/scratchpad/r5/`: `run.py`, `run_multi.py`, `drive.sh`, `multi.sh`, `all.sh`, `ana.py`, `out/*.json`. Rerun with `flock /tmp/wpjudge-gpu.lock ./all.sh`, then `ana.py -v`.
- Reference run `ref` = `--parallel 4`, workers=1, run 1.

## Evidence
| Config (server / client) | Runs | Units/min | Units byte-identical across runs | Max per-unit overall spread | Distinct `cap_reasons` sets | Verdict flips | Uncapped overall | vs `ref`: identical units, max \|Δoverall\| |
|---|---|---|---|---|---|---|---|---|
| `-np 4` / w1 | 3 | 5.4 | 10/10 | 0 | 1 | 0 | 68.8 ×3 | 10/10, 0 |
| `-np 4` / w1, `cache_prompt:false` | 2 | 5.4 | 10/10 | 0 | 1 | 0 | 68.8 ×2 | 10/10, 0 |
| `-np 1 -c 8192` / w1 | 3 | 5.4 | 10/10 | 0 | 1 | 0 | 68.8 ×3 | 10/10, 0 |
| **`-np 1 -c 8192` / w4** | 2 | **5.5** | **10/10** | **0** | 1 | 0 | 68.8 ×2 | **10/10, 0** |
| 3 × `-np 1` servers / w3 (round-robin) | 3 | 4.7 | 10/10 | 0 | 1 | 0 | 68.8 ×3 | 10/10, 0 |
| `-np 4 -kvu` / w1 | 1 | 5.5 | n/a | n/a | n/a | n/a | 68.3 | **8/10**, 2 |
| `-np 4` / w4 (today's default) | 3 | 10.0 | 0/10 | **28** | 3 | 0 | 68.9 / 70.3 / 69.4 | 0/10, **32** |
| `-np 4` / w4, `cache_prompt:false` | 3 | 9.8 | 0/10 | 10 | 2 | 0 | 70.1 / 70.1 / 68.6 | 0/10, 32 |
| `-np 4` / w4, `seed:42, top_k:1` | 2 | 10.5 | 0/10 | 10 | 2 | 0 | 68.5 / 68.5 | 0/10, 6 |
| `-np 4 -kvu` / w4 | 3 | 10.7 | 0/10 | 6 | 2 | 0 | 70.0 / 70.0 / 70.3 | 0/10, 32 |
| `-np 4 -nocb` / w4 | 2 | 8.4 | 0/10 | 10 | 2 | 0 | 70.8 / 68.4 | 0/10, 28 |

Pooled results for all 13 concurrent (w4 on `-np 4`) runs against `ref`:
- **No unit was byte-identical to `ref` in any run.** The first differing character is at offset 0 at minimum, 133 at the median and 851 at most. Batched prefill already changes the first token.
- Per-unit overall scores, with the `ref` value first:

  | Unit | `ref` | Concurrent runs |
  |---|---|---|
  | `ser_render_search_box` | 14 | 18 ×3, 42 ×5, 46 ×5 |
  | `ser_ajax_export_registrations` | 46 | 42–52 |
  | `ser_handle_cancel_registration` | 80 | 74–80 |
  | `ser_save_settings` | 46 | 46 ×13 (constant) |

- **Cap:** `capped` and the capped overall (60) never flipped on this fixture. Other units always cap it.
  - The `cap_reasons` set of (id, dim) matched `ref` in only 5/13 runs, and (id, dim, score) in **0/13**.
  - `security` moved 1↔4 and 3↔4, and `sql_safety` moved 2↔4.
  - `sql_safety` for `ser_ajax_export_registrations` was 4, 5, 7 or absent. Absent drops it from the cap set entirely.
- **Verdicts:** 0 flips in 130 unit-runs. This matches R2 (16/16).
- **Uncapped overall:** 68.4���70.8, against 68.8 for `ref`.

What the runs show:
1. **Worker count and slot count do not matter by themselves. Concurrent decoding does.**
   - `-np 4` with w1 equals `-np 1` with w4, byte for byte.
   - The server's queue serializes the four client workers, so the report is worker-invariant.
2. **The prompt cache is not the cause.** Sequential runs 2 and 3 reuse the cache and are still identical. `cache_prompt:false` leaves w4 just as noisy.
3. **Sampling is not the cause.** Temperature 0 is already greedy. A fixed seed with `top_k:1` changes nothing structurally.
4. **`-kvu` (unified KV) does not help w4, and it changes the sequential answer.** It differs from `ref` on 2/10 units at w1 because of a different KV layout (`n_ctx_slot` 32768). Do not use it.
5. **`-nocb` does not help.** Concurrent slots still share a batch, and throughput drops.
6. **Several single-slot servers are deterministic but slower than one server** (4.7 vs 5.4 units/min). They also use about 49 GB more memory (used memory went from 6 to 55 GB). The GPU time-shares and gives no overlap. Rejected.
7. **N-sample voting does not fix it.**
   - The median of 3 concurrent runs still varied per unit by [6, 4, 10, **28**, 0, 0, 2, 4, 2, 2] across the 286 triples.
   - That median matched `ref` in only 62% of unit-triples.
   - Voting also costs 3× the requests, which is worse than serializing. It is only worth it as a confidence signal (R4), and then it should run at `-np 1` (k× cost).

## Options
| Option | Worker-invariant? | Throughput | Verdict |
|---|---|---|---|
| A. Server `--parallel 1`, client `workers=1` default | **Yes**, for any client worker count and any other traffic on that server | 5.4 units/min (−46%; fixture 60 s → 111 s; R2's 300-function estimate 27 → ~55 min) | **Recommend** |
| B. Client `workers=1` only; server stays `-np 4` | Only if nothing else hits the server at the same time | Same as A | Weaker: a shared server or a user's `--workers 4` breaks it |
| C. `cache_prompt:false` / seed / `top_k:1` / `-kvu` / `-nocb` | No (table above) | About the same | Reject |
| D. N separate `-np 1` servers | Yes | 4.7 units/min, +~16 GB per copy | Reject |
| E. N-sample voting at `-np 4` | No (62% agreement) | ÷N | Reject for determinism |
| F. Keep `-np 4` and bound the noise | No. Bound on this fixture: \|Δunit\| ≤ 32, \|Δuncapped\| ≤ 2.4, `cap_reasons` scores differ in every run, verdicts 0/130 | 10.0 | Keep only as an opt-in fast mode that the report labels |

## Recommendation (smallest fix)
1. **Serialize by default.** Change the server default to `LLAMA_PARALLEL` 1, which also sets `WPJUDGE_WORKERS` 1 through compose. Change the client default to `workers` 1. Change the e2e host default to `HOST_PARALLEL` 1.
2. **Surface the opt-in fast mode.** Add `model.reproducible = (workers == 1 or server_slots == 1)` to the report.
   - `server_slots` comes from `GET {base_url without /v1}/props` → `total_slots`, which this llama.cpp build serves (`server-context.cpp:4527`).
   - If `/props` fails, `server_slots` is `null` and `reproducible = workers == 1`.
   - If `reproducible` is false, print one stderr warning.
   - This is an additive field, so `schema_version` stays "1.0".

   Residual hole: a multi-slot server shared with other clients while workers=1. Document it: "point wpjudge at its own `--parallel 1` server for reproducible reports".
3. Reproducibility is **per build and per server config**, not across them:
   - `-kvu` changed 2/10 units.
   - The README already notes that container b11515 differs from host 9954.
   - Goldens compare runs on the same server only.

## Files to change
- `wpjudge/config.py`: `workers: int = 4` → `1`.
- `docker/llama-entrypoint.sh`: `P=${LLAMA_PARALLEL:-4}` → `:-1`.
- `docker-compose.yml`: both `${LLAMA_PARALLEL:-4}` → `:-1`. Line 8 is the server, line 33 is `WPJUDGE_WORKERS`.
- `wpjudge/judge.py` (or `__init__.scan`): one `/props` probe that returns `total_slots` or `None`.
- `wpjudge/aggregate.build_report`: add `model.workers`, `model.server_slots` and `model.reproducible`.
- `wpjudge/cli.py`: print the warning when `reproducible` is false.
- `tests/e2e.sh`:
  - `HOST_PARALLEL=${HOST_PARALLEL:-1}`.
  - `cmd_golden`: replace "workers=4 vs workers=1: verdicts equal" and the dir-vs-zip/tgz "max |d| ≤ 6" with exact equality (see Tests).
- `README.md`:
  - Lines 33/36: `--parallel 1 -c 8192`, and drop "Keep `--workers` equal to `--parallel`".
  - Lines 51/130/150: defaults become 1.
  - Add a "Fast mode (`LLAMA_PARALLEL=4`, `--workers 4`): about 1.85× faster, scores wobble (≤ ±32 per unit, verdicts stable), `model.reproducible=false`" note.
  - Line 276: the host numbers now use `--parallel 1`.

## Tests
**Unit tests (no GPU)**
- `test_config`: `load().workers == 1`. `docker/llama-entrypoint.sh` contains `LLAMA_PARALLEL:-1`, and `docker-compose.yml` has no `LLAMA_PARALLEL:-4`.
- `test_aggregate`: `reproducible` is True for (1, 4), (4, 1) and (1, None), and False for (4, 4) and (4, None).
- `test_judge`: a fake client returns text keyed on the prompt, with random `sleep` jitter. `judge_units` at workers=1 and workers=4 returns identical lists. This guards against client-side order dependence; the existing `test_judge_units_sorted_and_callback` covers ordering only.
- `test_judge`: the `/props` probe returns `total_slots`. On connection error or a missing key it returns `None`.

**E2E**
- `golden` (`HOST_PARALLEL=1`): `r_dir` (w1) and `r_w4` (w4) must match exactly on:
  - per unit: `overall_score`, `scores`, `verdict`;
  - `summary`: `overall_score`, `overall_score_uncapped`, `cap_reasons`;
  - `dimensions`.
  - Make dir vs zip vs tgz exact too. Both pairs held 10/10 in this study (p1/w4 vs p4/w1 vs p1/w1).
  - Both reports have `model.reproducible == true`.
- New `determinism` section:
  - Run w4 against a `--parallel 4` host server. Assert `model.reproducible == false` and the stderr warning. Make no score assertion, because the output is nondeterministic.
  - Run w1 against the same server. Assert it equals the `golden` `r_dir` exactly, which shows the slot count alone does not move scores.
  - Cost: about 3 min of GPU.

## Risks
- **Throughput −46%.** A 300-function plugin takes ~55 min instead of ~27. Fast mode stays available, with a label.
- **Deterministic is not the same as more accurate.** The canonical greedy path gives `ser_render_search_box` 14 / security 1, while 10/13 concurrent runs gave 42–46 / security 4. Pinning one path can pin an outlier. That is R4's problem (confidence/voting); voting should run at `-np 1`.
- **The canonical output depends on build and flags** (`-kvu`, the build, and probably `-ub`/`-fa`). Pin the server flags in the entrypoint, and never compare goldens across host and container.
- **Fixture size:** 10 units, 14 sequential runs (all identical) and 13 concurrent runs. `cap`/`verdict` never flipped here, but they can on other targets: `cap_reasons` scores moved across the threshold (4↔5↔7 and absent).
- **External shared servers** with more than one slot and other traffic are still nondeterministic even at workers=1. `reproducible` only sees this server's slot count.
