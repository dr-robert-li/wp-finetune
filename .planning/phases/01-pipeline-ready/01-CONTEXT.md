# Phase 1: Pipeline Ready - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Harden all pipeline scripts for safe execution at scale and convert existing curated CSV data into repos.yaml. No data is generated or processed in this phase — the goal is a pipeline that won't waste API spend on failures. Two plans: (1) pipeline hardening, (2) CSV-to-repos.yaml conversion.

</domain>

<decisions>
## Implementation Decisions

### Repo Selection Criteria
- Filter from `wp_top1000_plugins_final.csv` (1,000 plugins) and `wp_top100_themes_final.csv` (100 themes)
- **Inclusion criteria:** Must have `github_url` (non-empty) — scripts need git clone access
- **Plugin filtering:** active_installs >= 10,000 AND rating_pct >= 80 AND unpatched_vulns == 0
- **Theme filtering:** active_installs >= 10,000 AND rating_pct >= 80 AND unpatched_vulns == 0
- **quality_tier assignment:** WordPress Core → "core" (auto-passed). Plugins/themes with 0 total_known_vulns AND rating >= 90 → "trusted". Everything else → "assessed" (full PHPCS + Claude judgment)
- **path_filters:** Auto-generate from tags column where possible (e.g., "page builder" → include builder-related PHP files). Default to `["*.php"]` with standard exclusions (vendor/, node_modules/, tests/)
- **Target count:** ~50-100 repos (enough for diversity without excessive API cost). If filtering produces >100, rank by active_installs and take top 100
- WordPress Core added manually as first entry (not in CSV data)

### Batch API Strategy
- **Hybrid approach:** Keep direct API calls for small runs (<50 items) with exponential backoff. Switch to Batch API for bulk operations (Phase 1 judge: ~5,000 calls, Phase 2 generate: ~2,000 calls)
- **Implementation:** Add a `batch_or_direct()` utility that checks item count and routes accordingly
- **Batch API specifics:** Submit JSONL batches, poll for completion, parse results. 24-hour window is fine for offline pipeline work
- **Direct API calls:** Replace fixed `time.sleep(REQUEST_INTERVAL)` with exponential backoff + jitter (base 1s, max 60s, factor 2x)
- **Rate limit handling:** Catch 429 responses, extract `retry-after` header, wait accordingly

### Checkpoint Granularity
- **Per-file checkpointing** for judgment phases (phase1_judge, phase2_judge, phase2_judge_dataset) — these make expensive API calls per function
- **Per-repo checkpointing** for clone and extract phases — these are cheap and fast
- **Implementation:** Write a checkpoint file (`{phase}_checkpoint.json`) tracking processed items. On restart, load checkpoint and skip completed items
- **Checkpoint format:** `{"completed": ["repo1/func1.php", "repo2/func2.php"], "failed": ["repo3/func3.php"], "timestamp": "..."}`
- **Batch API checkpoints:** Track batch job IDs so interrupted runs can poll existing batches instead of resubmitting

### Parse Failure Handling
- **Extract JSON parsing to shared `utils.py`** — single robust implementation used by all scripts
- **Parse strategy:** Try `json.loads()` on full response first. If fails, extract from markdown code blocks (```json...```). If fails, try regex for `{...}` blocks. If all fail → reject
- **On parse failure:** Log full response text to `{phase}_parse_failures.jsonl` for debugging. Do NOT create stub responses — reject the example entirely
- **Retry policy:** One retry with explicit "Return ONLY valid JSON, no markdown" instruction appended. If second attempt fails, reject
- **Existing output audit:** Pre-flight script scans existing output directories for stub responses (verdict: "FAIL" with no scores or empty critical_failures) and flags them for re-processing

### Claude's Discretion
- Exact exponential backoff parameters (base delay, max delay, jitter range)
- Checkpoint file location (alongside output or in dedicated checkpoints/ dir)
- Batch API polling interval
- Pre-flight script output format (table, JSON, or plain text)
- Whether to use `tenacity` library or hand-rolled retry logic

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pipeline Scripts (to be hardened)
- `scripts/phase1_clone.py` — Repo cloning logic, needs checkpoint support
- `scripts/phase1_extract.py` — PHP extraction, needs checkpoint support
- `scripts/phase1_judge.py` — PHPCS pre-filter + Claude judge, needs Batch API + checkpointing + robust parsing
- `scripts/phase2_generate.py` — Synthetic generation, needs Batch API + checkpointing
- `scripts/phase2_judge.py` — Synthetic judgment, needs Batch API + checkpointing + robust parsing
- `scripts/phase2_judge_dataset.py` — Judge dataset creation, needs rate limiting fix + Batch API + robust parsing
- `scripts/phase2_mutate.py` — Mutation engine, needs PHPCS availability check (hard-exit if missing)
- `scripts/phase3_cot.py` — CoT reasoning, needs Batch API + checkpointing

### Configuration
- `config/repos.yaml` — Target output for CSV conversion script
- `config/judge_system.md` — Judge system prompt (read-only reference for understanding expected JSON output format)

### Source Data
- `/home/robert_li/Desktop/data/wp-finetune-data/wp_top1000_plugins_final.csv` — 1,000 ranked plugins with github_url, active_installs, rating, vulnerability data
- `/home/robert_li/Desktop/data/wp-finetune-data/wp_top100_themes_final.csv` — 100 ranked themes with github_url, rating, vulnerability data

### Known Issues
- `.planning/codebase/CONCERNS.md` — Documents all known code issues (rate limiting gaps, JSON parsing fragility, etc.)

### Research
- `.planning/research/PITFALLS.md` — Pipeline-specific pitfalls and prevention strategies
- `.planning/research/STACK.md` — Anthropic Batch API details, recommended library versions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/phase1_judge.py` has existing rate limiting pattern (`REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE`) — can be refactored into shared utility
- `config/repos.yaml` has existing schema (url, quality_tier, path_filters, description) — CSV converter targets this exact format
- JSON parsing pattern exists in 4 scripts (identical code) — consolidate into `utils.py`

### Established Patterns
- All scripts use `anthropic.Anthropic()` client initialization from env var
- Output directories follow `{phase}_extraction/output/` or `{phase}_synthetic/output/` convention
- Scripts use `json.dumps()`/`json.loads()` for all data interchange
- Rate limiting via simple `time.sleep()` — needs upgrade to exponential backoff

### Integration Points
- `scripts/phase2_mutate.py` calls `phpcs` via subprocess — pre-flight must verify this works
- All judge scripts share identical JSON extraction logic — single point of consolidation
- `config/repos.yaml` is read by `phase1_clone.py` and `phase1_extract.py` — converter must match expected schema exactly

</code_context>

<specifics>
## Specific Ideas

- User has pre-curated CSV datasets at `/home/robert_li/Desktop/data/wp-finetune-data/` — these are the authoritative source for repo selection, not manual curation
- Vulnerability data (total_known_vulns, unpatched_vulns, max_cvss) should drive quality_tier assignment automatically
- User trusts Claude's judgment on implementation details — all gray areas resolved with sensible defaults

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-pipeline-ready*
*Context gathered: 2026-03-26*
