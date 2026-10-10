# Locked plan review (independent evaluator)

**Verdict: APPROVE WITH CHANGES.** Spec-faithful and small; research folded in correctly. Gaps: contract seams between workstreams, one impossible golden, no server parity check. All fixes are small.

## Required changes

1. **Fix the `worst_functions` golden.** ~9 judged fixture units vs top-10 list: every clean function is always listed. Assert instead that each planted unit scores below every clean unit.
2. **Make the cap golden robust to how the model scores.** `sql_safety` appears in 10/120 JSONs, so the cap rests on `security` ≤ 4, which is not guaranteed. Before locking the goldens, W6 runs each planted function once through the live judge and records its scores. The goldens assert (a) the harness invariant: `capped` == any judged unit has `security` or `sql_safety` ≤ 4, with exactly those units in `cap_reasons`, and (b) the planted SQL injection triggers the cap only if the pre-run showed it does. If none scores ≤ 4, strengthen the fixture, not the harness.
3. **Make the goldens deterministic.** Run the folder, .zip and .tar.gz golden scans with `workers=1`, which A4 shows is byte-identical. Add one `workers=4` run that asserts only units, skips and verdicts match. Assertions on scores must use tolerances (±6 per A4) or order relations, never equality.
4. **Define the skip rows in `functions[]`.** A skip row is `{file, line, name, loc: 0, truncated: false, status: "skipped", reason, verdict: null, overall_score: null, scores: {}, critique: ""}`. `reason` is the one added field. A judged or unjudged row is the schema fields plus `id, kind, end_line, attempts, error`. **Drop `code` along with `raw`.** (not in schema; ~19k bodies on woocommerce).
5. **Align the rank output with the schema.** The r2 `rank` uses `j["id"]`, `j["critique"]` and `j["scores"]`, and all three exist in Judgment, so it works. Two fixes:
   - It emits `examples: ["<id>: <reason>"]`, but the schema wants `"file:line fn"`. Emit `f'{file}:{line} {name}'`.
   - It also adds `examples` to `patterns`, which the schema does not have. Drop it there.
   
   Pass rank only judged rows.
6. **Move `DIM`, `TOPICS` and `rank` into `aggregate.py`.** Only W3 uses them. Removes the W3→W2 import during the parallel build.
7. **Give the unowned schema fields owners:**
   - `model.file` and `model.base_url` are filled by `wpjudge.scan()` (W4) from config.
   - Compose must pass `WPJUDGE_MODEL_FILE` to the **harness** service too. The current r4 draft sets it on llama only, so the report would name the wrong file.
   - `target.files_scanned` is the count of `.php` files walked, so `collect_units` must return it (W1). `load_target` cannot know it.
   - `coverage.units` = judged + unjudged. `pct` = judged / units × 100.
   - `cap_reasons` = `[{id, dimension, score}]`.
8. **Guard against zero judged units in W3.** With 0 judged, `overall_score` and `overall_score_uncapped` are `null` and `low_coverage` is true; no division by zero. Add to phase 5 a **server-down E2E**: point at a dead port, then expect exit 0, every unit unjudged, `low_coverage: true`, and a valid report. (R2.3's only real test).
9. **Disable hidden retries in W2.** Construct the client as `OpenAI(..., max_retries=0)`. The SDK default of 2 silently turns "retry once" into up to 6 calls.
10. **Pre-create the shared files before the parallel build.** The orchestrator writes `pyproject.toml` with all deps, `wpjudge/__init__.py` as a stub, and `wpjudge/config.py` with every knob: thresholds, MAX_CHARS, limits, workers, base_url, model file, templates, jobs dir, retention. Workstreams then only read them. (r1 had W1 editing pyproject too.) Fixtures: W1 owns `tests/fixtures/extract/` (edge.php, malicious archives), W2 owns `tests/fixtures/responses/`, and W6 owns `tests/fixtures/vuln-plugin/`.
11. **Fix the Docker build context.** The r4 draft uses `context: .` with `COPY harness/`, which assumes the compose file is at the repo root. The plan puts `docker-compose.yml` in `harness/`. Use `context: .` with `COPY . /app/` throughout.
12. **Add trust-boundary guards:**
    - Compose publishes `127.0.0.1:8000:8000` by default, because auth is off when the key is unset and `webhook_url` is an SSRF vector.
    - Accept `webhook_url` only with an `http` or `https` scheme.
    - Check job ids with `re.fullmatch(r"[0-9a-f]{32}", jid)`. `uuid.UUID(hex=)` also accepts braces and the `urn:uuid:` prefix.
    - Turn a non-numeric `Content-Length` into 400, not 500.
    - The CLI extracts archives into a `tempfile.TemporaryDirectory`.
    - Phase 5 tests `/jobs/..%2f../functions` and auth on `/functions`.
13. **Add a model-server parity check to phase 5** (study: host build 8f114a9b5; container: b11515). r2's 16 prompts, `workers=1`, both servers: 16/16 verdicts equal, `overall_score` |Δ| ≤ 6. Driver-580 smoke test first.
14. **Bound the phase 5 runtime and pick small real targets.** At about 11 functions/min, use `classic-editor` (39 units, about 4 min) and `twentytwentyfour` (60 units, mostly templates). Not cf7 (~70 min) or woocommerce (~29 h). The CLI prints the unit count and an estimate (units ÷ 11/min) to stderr before judging.
15. **Document the out-of-distribution templates.** Templates can dominate a theme's LOC-weighted mean. Phase 5 records the theme's template vs function score split. The README states the caveat and `WPJUDGE_TEMPLATES=0`. Keep the default on, per R2.2.
16. **Add the CLI flags A4 needs:** `--workers`, `--base-url` and `--model-file`, plus `-o`.

## Optional suggestions
- Webhook via stdlib `urllib.request`; httpx stays a test extra.
- `allinone.sh` sources the download block from `llama-entrypoint.sh`.
- `judge_units`: `submit` + `as_completed` so progress is not blocked by one slow unit; sort output by `id`. Tie-break `worst_functions` by `id`.
- Drop the overflow-halving retry if phase 5 never overflows at 15000 chars.
- README: one API job blocks the queue for hours on large plugins; no cancel in v1.
