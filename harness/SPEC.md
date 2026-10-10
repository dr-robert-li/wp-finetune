# WP Judge Harness — Spec (draft, built by Q&A)

Goal: an operator-ready package. Input: a WordPress plugin or theme (folder, .zip or .tar.gz). Output: one JSON
report. Each PHP function is judged by the v4 judge GGUF. Scores roll up to an overall score with
deviations, patterns and anti-patterns. Use as a Python library or CLI in-session, or as a persistent container API.

Rule: self-contained in `harness/`. Copy existing repo assets in and modify them. Build new code only where nothing exists.
Component inventory: `INVENTORY.md`.

## Decisions

| # | Topic | Decision |
|---|---|---|
| R1.1 | Model host | **Both.** Default: docker-compose with the harness (thin client) + the official llama.cpp server image, GGUF on a mounted volume. Optional all-in-one Dockerfile target (llama-server + harness in one image, GGUF mounted or baked). The harness only needs an OpenAI-compatible base URL. |
| R1.2 | API mode | **Async jobs.** POST upload → `job_id`. GET `/jobs/{id}` → progress, then the report. In-process worker. Results on disk. No Redis or Celery. |
| R1.3 | Aggregation | **LOC-weighted mean of overall_score + safety cap.** The cap trigger is defined in R2.1 (this row's original "any FAIL" trigger is superseded). Report a per-dimension mean, std dev, FAIL rate and the worst functions. |
| R1.4 | Patterns | **Model output only.** Rank recurring per-dimension FAILs and critique findings across functions by frequency, plus strengths. No rule engine and no PHPCS in v1. |
| R2.1 | Verdict + cap | Report the **model's verdict** unchanged. Do not recompute it. Cap trigger: any function with `security` or `sql_safety` ≤ 4/10. The plugin overall is then capped at 60 and flagged. Thresholds are in config. Send no system prompt. User message: `<wp_judge> Evaluate this WordPress code:` + a php fence. |
| R2.2 | Scope | Judge all functions and methods. Skip tests, `vendor/`, `node_modules/`, build/dist, minified files and functions under 3 lines. Judge a template file with top-level PHP and no functions as one unit, if under the size limit. Truncate over-long functions and flag them. |
| R2.3 | Failures | Retry once. Then mark the function unjudged and exclude it from the score. Report coverage %. Flag the report "low coverage" if coverage is under 90%. |
| R2.4 | API surface | All four: bearer API key from an env var (auth is off if unset); upload limits (50 MB default, zip-bomb and traversal guard) and job retention (7 days default); optional webhook POST of the report on completion; `GET /jobs/{id}/functions` for partial per-function results. |
| R3.1 | GGUF delivery | Download on first run: the entrypoint runs `hf download` into a named volume if the file is missing. The repo is public, so no token is needed. The env var `WPJUDGE_MODEL_FILE` selects the file. Default: `wp-judge-v4-pruned-k224.IQ4_XS.gguf`. |
| R3.2 | Concurrency | llama-server `--parallel 4`, ctx 4×8192. The harness sends 4 concurrent requests. Both settings are configurable. Greedy decoding (temperature 0), `max_tokens` 2048, thinking off. |
| R3.3 | Packaging | One package `harness/wpjudge/` (modules: extract, judge, aggregate, api) + `pyproject.toml`. CLI: `wpjudge scan <dir|zip|tar.gz> -o report.json` and `wpjudge serve`. Dependencies: openai, fastapi, uvicorn, python-multipart. php-cli in the image. |

## Amendments after research (2026-10-10; evidence in research/)
| # | Amends | Change |
|---|---|---|
| A1 | R3.3 | Extraction uses **tree-sitter + tree-sitter-php** (pip, aarch64 wheels). **No php-cli** in the image or on hosts. The PHP extractor is not copied: it mis-extracts 16/83 files in contact-form-7 (r1). |
| A2 | R3.1 | The model download uses **`curl -fL -C -`** on the HF resolve URL (resumable; curl is already in the llama image), not `hf download` (r4). |
| A3 | R2.2 | New skip reason `too_large`: a template over the size limit. Per-file skips use `line: 0, name: ""`. Truncation limit: 15000 chars (r1). |
| A4 | R3.2 | 4 slots ≈ 1.9× throughput, but parallel output is **not byte-deterministic** (verdicts are stable; overall_score moves by up to ±6). The default stays at 4. `workers=1` gives a reproducible report. Document this (r2). |
| A5 | R1.4 | Scores come **only from `<judge_output>`** (41% of the prose lines carry wrong scores). Patterns use JSON score ≥ 8, anti-patterns ≤ 4. The reason clause is tagged with a fixed 16-topic lexicon and counted once per function, with one example kept (r2). |
| A6 | R1.1 | The llama service uses `ghcr.io/ggml-org/llama.cpp:server-cuda13-b11515` (arm64 + sm_121 SASS). Never use `server-cuda` (12.8, no sm_121) (r4). |

## Report schema (v1, accepted)
```json
{
  "schema_version": "1.0",
  "target": {"name": "", "type": "plugin|theme", "files_scanned": 0, "sha256": ""},
  "model": {"file": "", "base_url": ""},
  "summary": {
    "overall_score": 0, "overall_score_uncapped": 0, "capped": false, "cap_reasons": [],
    "verdict_counts": {"PASS": 0, "FAIL": 0},
    "coverage": {"units": 0, "judged": 0, "unjudged": 0, "skipped": 0, "pct": 0.0, "low_coverage": false}
  },
  "dimensions": {"security": {"mean": 0, "std": 0, "min": 0, "n_rated": 0, "low_rate": 0.0}},
  "anti_patterns": [{"dimension": "", "finding": "", "count": 0, "examples": ["file:line fn"]}],
  "patterns": [{"dimension": "", "finding": "", "count": 0}],
  "worst_functions": [{"file": "", "line": 0, "name": "", "overall_score": 0, "verdict": ""}],
  "functions": [{"file": "", "line": 0, "name": "", "loc": 0, "truncated": false, "status": "judged|unjudged|skipped",
                  "verdict": "", "overall_score": 0, "scores": {}, "critique": ""}]
}
```

## Next
- Confirm the report schema; then write the build plan (copy list from INVENTORY.md + greenfield gaps).
