# EVAL v1.1 WS-C: T2b critique sync + T4 confidence

Evaluator: independent (I did not build this). Worktree `agent-a1ae90a2d758cc840`, HEAD c4f3a5d, diff `5e5ea1a..HEAD`.
Spec: PLAN_v1.1.md WS-C and D2. Research: R2 §b, R4. Builder report: `build_notes/V11_WSC.md`.
I edited no code. The only setup step was copying the gitignored `tests/fixtures/vuln-plugin/build/` from main into the worktree.

## Spec conformance
| Spec item | Status |
|---|---|
| Both `_judge_create` calls send `logprobs=True`, with no `top_logprobs` | done (`judge.py:26,39`); unit test covers both paths |
| `verdict_p` = exp(logprob) of the first non-blank, non-quote token after `"verdict":` inside `<judge_output>`; read with `getattr`; None when there are no logprobs or no verdict | done (`verdict_prob`); rounded to 3 places, as R4 says |
| `to_judgment(unit, raw, attempts, verdict_p=None)` stays cfg-free, stores `prose_scores`, rewrites headers to `Label: score <json or None>/10`, leaves `raw` unchanged | done (`_sync_headers` reuses `aggregate.DIM`; no circular import) |
| `_row`: `verdict_p`, `prose_scores`, `confidence_reasons` (`verdict_p` / `prose_disagrees`, computed from cfg when the report is aggregated), `low_confidence` | done |
| `prose_disagrees`: different sides of `cfg.cap_threshold` for any `cfg.cap_dims`; None, N/A or absent counts as above | done |
| `summary.low_confidence`; skip and unjudged rows get false, `[]` and null | done (checked live on serverdown: unjudged row has `verdict_p` null, `[]`, false; summary 0) |
| `config.min_verdict_p = 0.9`, `WPJUDGE_MIN_VERDICT_P`, where 0 disables | done |
| Unit tests: R2 sync and cap-ignores-prose; R4 extraction (split tokens, no logprobs, template path); confidence cases; skip-row exact dict | done; the tests are meaningful (see below) |
| E2E chk: header sync, the exact 5 planted cap ids (hard on cli, soft on api/aio), `0 ≤ verdict_p ≤ 1`, summary count, `MIN_VERDICT_P=0`, classic-editor soft `prose_disagrees`, low_confidence INFO | done |

## Findings
1. **MINOR: headers without a `— reason` tail are not synced.** `_sync_headers` uses `DIM`, which needs the dash and reason tail. A bare `Security: severity high (4/10)` line would keep its stale number. The plan accepts this for the e2e check. Evidence: across all v1 saved reports (`build_notes/e2e_work/**/*.json`), 1789 header lines match the header prefix and **0** of them lack the tail. Fix (optional): none needed now. If it ever shows up, give `_sync_headers` R2's tail-free `HDR` regex.
2. **MINOR: `logprobs=True` is sent unconditionally.** A server that rejects the parameter outright, instead of returning null, raises an error that the template fallback does not catch, so every unit becomes unjudged. The plan requires this. llama-server (build 9954) accepts it. Fix: WS-F should add a README line next to R4's "non-llama.cpp servers may return no logprobs" caveat.
3. **MINOR (process): `e2e.sh` lines added outside the owned `cmd_fixture`.** These are one `chk fixture` line in `sec_cli` and a `confidence()`/classic-editor block in `cmd_real`. They are needed for the plan's "hard on host cli" and classic-editor checks, and the builder declared them. They may conflict with B (`cmd_real`) and D (`sec_cli`) at merge.
4. **INFO: precision of the flag.** classic-editor gives `summary.low_confidence` 9/39 (23%):
   - all 9 rows have `prose_disagrees`; 8 of them are prose 4 vs JSON 6–7, and `settings_2` is prose 2 vs JSON 6;
   - one row also has `verdict_p` (`settings_2`, p 0.552);
   - `add_redirect_helper` is flagged as the plan wants: prose 4, JSON 6, p 0.995.

   This matches the plan's "about 25%", and the README (WS-F) has to say so. On the fixture, 3/10 rows are flagged:
   - two planted FAILs at p .887 and .891 get `verdict_p`. They are correct verdicts, as R4 predicted;
   - the clean `ser_handle_cancel_registration` gets `prose_disagrees` (prose 4, JSON 8). R2's FP case is now flagged and not capped.
5. **INFO:** the builder committed on the worktree branch, against the PLAN's "Do not commit". This follows the builder brief, which disagrees with the plan. It was declared.

No BLOCKER or MAJOR.

## Code review notes (no issues)
- `verdict_prob`:
  - it searches for `"verdict"` only after the first `<judge_output>`, so a `"verdict"` in the prose is ignored (tested);
  - a token that ends exactly at the colon is skipped;
  - a merged token such as `": "FA` takes that token's p, which is acceptable.
- `judge_one` passes the logprobs of the attempt that produced the judgment, so a retry does not mix attempts.
- `_row` overwrites `verdict_p` to None on unjudged rows, and `setdefault` handles judgment dicts written before WS-C.
- Unit tests are not performative:
  - the sync test checks the real `severity_format_9` fixture: prose 4 → `score 6/10`, SQL → `None/10`, no `severity` left, and `raw` unchanged;
  - the extraction tests cover the split-token and prose-decoy cases;
  - `prose_disagrees` is tested at `cap_threshold=5` in both directions.

## Commands and results
1. `.venv/bin/python -c 'import wpjudge;print(wpjudge.__file__)'` prints a path inside the worktree. `.venv/bin/python -m pytest tests/` gives **46 passed**.
2. Live e2e:
   ```
   flock /tmp/wpjudge-gpu.lock: GGUF=…/shrink-iq4xs-im_f16.gguf HOST_PARALLEL=1 WPJUDGE_WORKERS=1 PORT=8197 WORK=<scratch>/e2e tests/e2e.sh cli serverdown
   ```
   Result: **168 passed, 0 failed**, exit 0. The server was killed by the script, and no server on 8197 remains.
   - On r_dir, r_zip and r_tgz:
     - headers synced (bad []);
     - `verdict_p` null 0, min 0.887;
     - `low_confidence` 3/10;
     - the `MIN_VERDICT_P=0` check passed;
     - `cap_reasons` ids are exactly the 5 planted bugs (extra [], missing []).
   - golden dir/zip/tgz/w4 all passed.
   - r_dir overall: capped 60. Planted [36, 46, 46, 14, 46], clean [80, 80, 80, 84, 84].
   - I read the critique text myself: headers read e.g. `SQL Safety: score 2/10 — Issue: …`, and the reason tails are kept.
3. **Logprobs cost** (plan round-2 N3). Setup:
   - one host llama-server (build 9954), `--parallel 1 -c 8192`, port 8197, under flock;
   - classic-editor (`build_notes/e2e_work/real/classic-editor.zip`, 39 units) was scanned twice, `--workers 1`;
   - the base run used the package from `git archive 5e5ea1a harness/wpjudge` via PYTHONPATH (I confirmed `wpjudge.__file__` pointed at the base copy, which has no logprobs);
   - the WS-C run used the worktree venv;
   - timings are summed from the per-request `eval time` lines in the llama-server log.

   | Run | Requests | Completion tokens | Eval ms | **ms / completion token** | Wall |
   |---|---|---|---|---|---|
   | base 5e5ea1a (no logprobs) | 39 | 33052 | 382539 | **11.574** | 392 s |
   | WS-C (`logprobs=True`) | 39 | 32893 | 403486 | **12.266** | 414 s |

   - The cost is **+6.0% per completion token** (+5.6% wall), within the plan's +15% bound. It is lower than R4's +12%, which used `top_logprobs=5` at w4.
   - The outputs are identical: 0 of 39 units differ in overall, scores or verdict. Both overalls are 73.9, uncapped, coverage 100%.
   - classic-editor **`summary.low_confidence` / units = 9 / 39**. `verdict_p` is non-null on 39/39, min 0.552.
4. **Re-aggregation from cfg** (D2). I re-ran `build_report` on the saved reports with `WPJUDGE_CAP_THRESHOLD` = 4, 5 and 3 (`<scratch>/reagg.py`):

   | Report | T | Rows with `prose_disagrees` | `low_confidence` |
   |---|---|---|---|
   | fixture r_dir | 4 | 1: `handle_cancel` | 3 |
   | fixture r_dir | 5 | 1: `handle_cancel` | 3 |
   | fixture r_dir | 3 | 3: `delete_registration`, `ajax_export`, `save_settings` | 3 |
   | classic-editor | 4 | 9 | 9 |
   | classic-editor | 5 | 9 | 9 |
   | classic-editor | 3 | 1: `settings_2` | 1 |

   - Fixture at T=3: prose 2 vs JSON 4 now straddle the threshold. The prose 4 vs JSON 8 pair no longer does.
   - Classic-editor at T=3: prose 4 vs JSON 6 no longer straddle; `settings_2` is prose 2.
   - So the flag is recomputed from `cfg.cap_threshold` and `prose_scores` when the report is aggregated.
   - T=5 leaves both reports unchanged, and that is correct: neither source contains a 5, so no pair changes sides. The unit test `test_prose_disagrees` covers T=5 both ways: (5, 6) flags and (4, 5) does not.

## VERDICT: PASS
