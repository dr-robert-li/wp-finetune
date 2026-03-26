# Phase 1: Pipeline Ready - Research

**Researched:** 2026-03-26
**Domain:** Python data pipeline hardening + CSV-to-YAML conversion (no ML, no GPU)
**Confidence:** HIGH — all findings grounded in direct codebase audit and official Anthropic docs

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Repo Selection Criteria**
- Filter from `wp_top1000_plugins_final.csv` (1,000 plugins) and `wp_top100_themes_final.csv` (100 themes)
- Inclusion criteria: Must have `github_url` (non-empty) — scripts need git clone access
- Plugin filtering: active_installs >= 10,000 AND rating_pct >= 80 AND unpatched_vulns == 0
- Theme filtering: active_installs >= 10,000 AND rating_pct >= 80 AND unpatched_vulns == 0
- quality_tier assignment: WordPress Core → "core" (auto-passed). Plugins/themes with 0 total_known_vulns AND rating >= 90 → "trusted". Everything else → "assessed" (full PHPCS + Claude judgment)
- path_filters: Auto-generate from tags column where possible (e.g., "page builder" → include builder-related PHP files). Default to `["*.php"]` with standard exclusions (vendor/, node_modules/, tests/)
- Target count: ~50-100 repos (enough for diversity without excessive API cost). If filtering produces >100, rank by active_installs and take top 100
- WordPress Core added manually as first entry (not in CSV data)

**Batch API Strategy**
- Hybrid approach: Keep direct API calls for small runs (<50 items) with exponential backoff. Switch to Batch API for bulk operations (Phase 1 judge: ~5,000 calls, Phase 2 generate: ~2,000 calls)
- Implementation: Add a `batch_or_direct()` utility that checks item count and routes accordingly
- Batch API specifics: Submit JSONL batches, poll for completion, parse results. 24-hour window is fine for offline pipeline work
- Direct API calls: Replace fixed `time.sleep(REQUEST_INTERVAL)` with exponential backoff + jitter (base 1s, max 60s, factor 2x)
- Rate limit handling: Catch 429 responses, extract `retry-after` header, wait accordingly

**Checkpoint Granularity**
- Per-file checkpointing for judgment phases (phase1_judge, phase2_judge, phase2_judge_dataset) — these make expensive API calls per function
- Per-repo checkpointing for clone and extract phases — these are cheap and fast
- Implementation: Write a checkpoint file (`{phase}_checkpoint.json`) tracking processed items. On restart, load checkpoint and skip completed items
- Checkpoint format: `{"completed": ["repo1/func1.php", "repo2/func2.php"], "failed": ["repo3/func3.php"], "timestamp": "..."}`
- Batch API checkpoints: Track batch job IDs so interrupted runs can poll existing batches instead of resubmitting

**Parse Failure Handling**
- Extract JSON parsing to shared `utils.py` — single robust implementation used by all scripts
- Parse strategy: Try `json.loads()` on full response first. If fails, extract from markdown code blocks (```json...```). If fails, try regex for `{...}` blocks. If all fail → reject
- On parse failure: Log full response text to `{phase}_parse_failures.jsonl` for debugging. Do NOT create stub responses — reject the example entirely
- Retry policy: One retry with explicit "Return ONLY valid JSON, no markdown" instruction appended. If second attempt fails, reject
- Existing output audit: Pre-flight script scans existing output directories for stub responses (verdict: "FAIL" with no scores or empty critical_failures) and flags them for re-processing

### Claude's Discretion
- Exact exponential backoff parameters (base delay, max delay, jitter range)
- Checkpoint file location (alongside output or in dedicated checkpoints/ dir)
- Batch API polling interval
- Pre-flight script output format (table, JSON, or plain text)
- Whether to use `tenacity` library or hand-rolled retry logic

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PIPE-01 | Pipeline pre-flight script validates PHPCS install, API key, PHP CLI, and WordPress-Coding-Standards before execution | Pre-flight pattern documented; subprocess + env var checks needed; exact check commands verified |
| PIPE-02 | All long-running scripts support checkpoint/resume to survive interruptions | Checkpoint format specified; per-file vs per-repo granularity documented; CONCERNS.md confirms no checkpointing today |
| PIPE-03 | API calls use exponential backoff with jitter instead of fixed sleep intervals | All 4 scripts using fixed sleep confirmed; anthropic SDK retry behavior documented; hand-rolled pattern specified |
| PIPE-04 | Scripts integrate Anthropic Batch API for high-volume offline processing (50% cost savings) | Batch API JSONL format, polling, and result parsing confirmed in Anthropic docs; hybrid threshold at 50 items locked |
| PIPE-05 | Parse failure stubs are detected and rejected instead of silently entering training data | Identical brittle parsing in 3 scripts confirmed; shared `utils.py` consolidation approach specified |
| REPO-01 | repos.yaml populated with WordPress Core repository | Already present in `config/repos.yaml`; CSV converter must preserve it as first entry |
| REPO-02 | repos.yaml populated with 10+ high-quality plugins selected from ranked CSV | CSV schema confirmed; filter logic for 10,000+ installs / 80+ rating / 0 unpatched_vulns specified |
| REPO-03 | repos.yaml populated with 5+ high-quality themes selected from ranked CSV | CSV schema confirmed; same filter thresholds; theme-specific path_filter logic needed |
| REPO-04 | Each repo entry has quality_tier (auto-assigned), path_filters, and description | quality_tier logic fully specified; path_filter auto-generation from tags column documented |
</phase_requirements>

---

## Summary

Phase 1 has two independent workstreams. The first is pure refactoring and infrastructure: harden eight existing pipeline scripts by extracting a shared `utils.py`, adding pre-flight validation, per-file checkpointing, exponential backoff on direct API calls, and wiring in the Anthropic Batch API for bulk judge/generate phases. The second is a new conversion script that reads two pre-existing CSV files (1,000 plugins, 100 themes) and emits a fully-populated `config/repos.yaml` — replacing the currently manual and incomplete configuration.

All code in this phase is pure Python + standard library + the `anthropic` SDK. No ML, no GPU, no new external services. The risk is correctness, not complexity: checkpoint files must be written atomically to avoid corruption, the Batch API JSONL format must match the API spec exactly, and the CSV-to-YAML converter must preserve the existing `repos.yaml` schema consumed by downstream scripts.

The existing codebase is in known-good shape for logic; the problems are all infrastructure-level (no retry, no resume, no pre-flight, brittle JSON extraction). All scripts currently use `anthropic.Anthropic()` with the same client pattern, making the shared utility refactor straightforward.

**Primary recommendation:** Build `utils.py` first — it is the dependency for all other hardening tasks. The CSV converter is independent and can be developed in parallel.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| anthropic | >=0.50.0 (current in project) | Batch API + direct API calls | Only official Python SDK; Batch API requires `client.beta.messages.batches` namespace |
| pyyaml | >=6.0 (current in project) | Write repos.yaml | Already used by all scripts; safe_dump is idiomatic |
| csv (stdlib) | Python stdlib | Read CSV source data | No dep install; CSV files are straightforward tabular data |
| json (stdlib) | Python stdlib | Checkpoint files, parse failure logs | Already used everywhere |
| pathlib (stdlib) | Python stdlib | File paths | Already used everywhere |
| re (stdlib) | Python stdlib | Regex JSON extraction in parse fallback | Already used in scripts |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tenacity | >=8.0 | Exponential backoff decorator | Claude's Discretion — use if preferred over hand-rolled; NOT currently installed on this machine |
| time (stdlib) | Python stdlib | Hand-rolled retry sleep | Use if tenacity not installed; simpler to audit |
| subprocess (stdlib) | Python stdlib | Pre-flight checks (phpcs --version, php --version) | Already used in phase1_judge.py for PHPCS |

**Note:** `tenacity` is not installed in the local Python environment. Since it is Claude's Discretion, the plan should default to hand-rolled retry (10 fewer lines, zero new deps). If tenacity is desired, add `pip install tenacity` to setup notes.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled backoff | tenacity | tenacity is cleaner but adds a dependency; hand-rolled is transparent and already patterns in project |
| Per-file checkpoint JSON | SQLite | SQLite is faster at scale but overkill for ≤13,500 items; JSON is readable and debuggable |
| Batch API polling loop | asyncio | asyncio adds complexity; polling with `time.sleep(30)` is sufficient for 24-hour async window |

**Installation (if tenacity chosen):**
```bash
pip install tenacity
```

---

## Architecture Patterns

### Recommended Project Structure (additions only)

```
scripts/
├── utils.py                    # NEW: shared utilities (parse, backoff, checkpoint, preflight)
├── csv_to_repos.py             # NEW: CSV → repos.yaml converter
├── preflight.py                # NEW: standalone pre-flight check script
├── phase1_clone.py             # MODIFY: add per-repo checkpoint
├── phase1_extract.py           # MODIFY: add per-repo checkpoint
├── phase1_judge.py             # MODIFY: use utils.py, add per-file checkpoint, Batch API
├── phase2_generate.py          # MODIFY: use utils.py, add per-item checkpoint, Batch API
├── phase2_judge.py             # MODIFY: use utils.py, add per-file checkpoint, Batch API
├── phase2_judge_dataset.py     # MODIFY: add missing rate limiting, use utils.py, Batch API
├── phase2_mutate.py            # MODIFY: fail-hard on missing PHPCS (remove graceful degrade)
└── phase3_cot.py               # MODIFY: use utils.py, add per-file checkpoint, Batch API

checkpoints/                    # NEW dir (alongside project root or in .planning/)
├── phase1_judge_checkpoint.json
├── phase2_judge_checkpoint.json
└── ...
```

### Pattern 1: Shared `utils.py` — Robust JSON Extraction

**What:** Single function that tries multiple extraction strategies in order.
**When to use:** Every script that calls Claude and parses the response as JSON.

```python
# Source: direct codebase analysis — consolidates identical logic from
#         phase1_judge.py:195-200, phase2_judge.py:47-51, phase2_judge_dataset.py:70-74

import json
import re

def extract_json(text: str) -> dict | None:
    """
    Try multiple strategies to extract JSON from a Claude response.
    Returns parsed dict or None if all strategies fail.
    """
    text = text.strip()

    # Strategy 1: Full response is already valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: ```json ... ``` fence (most common)
    m = re.search(r'```json\s*([\s\S]+?)\s*```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: ``` ... ``` fence (no language hint)
    m = re.search(r'```\s*([\s\S]+?)\s*```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 4: Outermost { ... } block
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None
```

### Pattern 2: Exponential Backoff + Jitter for Direct API Calls

**What:** Replace all `time.sleep(REQUEST_INTERVAL)` with a retry wrapper.
**When to use:** Any direct `client.messages.create()` call in a judge or generate script.

```python
# Source: based on Anthropic rate limit guidance + standard backoff pattern
# https://platform.claude.com/docs/en/api/rate-limits

import time
import random
import anthropic

def call_with_backoff(
    client: anthropic.Anthropic,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    factor: float = 2.0,
    max_retries: int = 5,
    **kwargs
) -> anthropic.types.Message:
    """
    Call client.messages.create(**kwargs) with exponential backoff + jitter.
    Reads retry-after header on 429 if available.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            # Prefer retry-after header if available
            retry_after = getattr(e, 'retry_after', None)
            wait = float(retry_after) if retry_after else delay
            jitter = random.uniform(0, wait * 0.1)
            time.sleep(wait + jitter)
            delay = min(delay * factor, max_delay)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < max_retries - 1:
                jitter = random.uniform(0, delay * 0.1)
                time.sleep(delay + jitter)
                delay = min(delay * factor, max_delay)
            else:
                raise
```

### Pattern 3: Per-File Checkpoint

**What:** Write progress to `checkpoints/{phase}_checkpoint.json` after each completed item.
**When to use:** phase1_judge, phase2_judge, phase2_judge_dataset, phase3_cot.

```python
# Source: designed from scratch per CONTEXT.md checkpoint format spec

import json
from pathlib import Path
from datetime import datetime

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

def load_checkpoint(phase: str) -> dict:
    """Load existing checkpoint or return empty state."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"{phase}_checkpoint.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "batch_job_ids": [], "timestamp": None}

def save_checkpoint(phase: str, state: dict) -> None:
    """Atomically write checkpoint (write to tmp, rename)."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"{phase}_checkpoint.json"
    tmp_path = path.with_suffix(".tmp")
    state["timestamp"] = datetime.utcnow().isoformat()
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.rename(path)   # atomic on POSIX
```

### Pattern 4: Anthropic Batch API Submit + Poll

**What:** Submit a JSONL batch, poll for completion, yield results.
**When to use:** When item count >= 50 (locked decision). Used in phase1_judge, phase2_generate, phase2_judge, phase3_cot.

```python
# Source: Anthropic Batch API docs
# https://docs.anthropic.com/en/api/creating-message-batches

import time
import anthropic

def submit_batch(client: anthropic.Anthropic, requests: list[dict]) -> str:
    """
    Submit a message batch. Each request: {"custom_id": str, "params": MessageCreateParams}
    Returns batch_id.
    """
    batch = client.beta.messages.batches.create(requests=requests)
    return batch.id

def poll_batch(client: anthropic.Anthropic, batch_id: str, poll_interval: int = 30) -> list:
    """
    Poll until batch completes. Returns list of result objects.
    24-hour window — no timeout needed for offline pipeline.
    """
    while True:
        batch = client.beta.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)
    # Stream results
    results = []
    for result in client.beta.messages.batches.results(batch_id):
        results.append(result)
    return results

def batch_or_direct(item_count: int) -> str:
    """Route decision: 'batch' if >= 50 items, 'direct' otherwise."""
    return "batch" if item_count >= 50 else "direct"
```

### Pattern 5: Batch Request JSONL Format

```python
# Source: Anthropic Batch API — request object format

def make_batch_request(custom_id: str, system: str, user_content: str,
                       model: str = "claude-sonnet-4-6", max_tokens: int = 1024) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_content}],
        }
    }
```

### Pattern 6: Pre-Flight Check

```python
# Source: codebase analysis — PHPCS silent degradation documented in CONCERNS.md

import subprocess
import os
import sys

def run_preflight() -> None:
    """Exit with clear error if any required tool is missing."""
    checks = [
        ("ANTHROPIC_API_KEY", lambda: os.environ.get("ANTHROPIC_API_KEY")),
        ("php", lambda: subprocess.run(
            ["php", "--version"], capture_output=True).returncode == 0),
        ("phpcs", lambda: subprocess.run(
            ["phpcs", "--version"], capture_output=True).returncode == 0),
        ("WordPress-Coding-Standards", lambda: "WordPress-Extra" in subprocess.run(
            ["phpcs", "-i"], capture_output=True, text=True).stdout),
    ]
    failed = []
    for name, check in checks:
        try:
            if not check():
                failed.append(name)
        except FileNotFoundError:
            failed.append(name)
    if failed:
        print(f"ERROR: Pre-flight failed. Missing: {', '.join(failed)}", file=sys.stderr)
        print("Fix before running any pipeline script.", file=sys.stderr)
        sys.exit(1)
    print("Pre-flight: all checks passed.")
```

### Pattern 7: CSV-to-repos.yaml Converter Logic

**What:** Read CSV, apply filters, auto-assign quality_tier and path_filters, emit repos.yaml.
**When to use:** `scripts/csv_to_repos.py` — standalone conversion script.

```python
# Source: CONTEXT.md filter criteria + existing repos.yaml schema

import csv
import yaml

# CSV columns (confirmed from file header):
# rank, name, slug, active_installs, rating_pct, total_known_vulns, unpatched_vulns, github_url, tags

PLUGIN_CSV = "/home/robert_li/Desktop/data/wp-finetune-data/wp_top1000_plugins_final.csv"
THEME_CSV  = "/home/robert_li/Desktop/data/wp-finetune-data/wp_top100_themes_final.csv"
REPOS_YAML = "config/repos.yaml"

# Inclusion filters (locked)
MIN_INSTALLS  = 10_000
MIN_RATING    = 80
MAX_UNPATCHED = 0

# quality_tier logic (locked):
#   unpatched_vulns == 0 AND total_known_vulns == 0 AND rating_pct >= 90 → "trusted"
#   everything else → "assessed"

TAG_TO_PATH_FILTERS = {
    "page builder": ["includes/**/*.php", "modules/**/*.php", "widgets/**/*.php"],
    "seo":          ["includes/**/*.php", "src/**/*.php"],
    "woocommerce":  ["includes/**/*.php", "src/**/*.php"],
    "security":     ["includes/**/*.php", "src/**/*.php"],
    "e-commerce":   ["includes/**/*.php", "src/**/*.php"],
    # default fallback:
    "_default":     ["**/*.php"],
}

STANDARD_SKIP = ["vendor/", "node_modules/", "tests/", "test/", "assets/", "css/", "js/"]

def assign_quality_tier(row: dict) -> str:
    total = int(row.get("total_known_vulns") or 0)
    unpatched = int(row.get("unpatched_vulns") or 0)
    rating = float(row.get("rating_pct") or 0)
    if unpatched == 0 and total == 0 and rating >= 90:
        return "trusted"
    return "assessed"

def infer_path_filters(tags_str: str) -> list[str]:
    tags = [t.strip().lower() for t in (tags_str or "").split(",")]
    for keyword, filters in TAG_TO_PATH_FILTERS.items():
        if keyword != "_default" and any(keyword in t for t in tags):
            return filters
    return TAG_TO_PATH_FILTERS["_default"]
```

### Anti-Patterns to Avoid

- **Silent PHPCS degradation:** `except FileNotFoundError: return True` in phase2_mutate.py — change to `sys.exit(1)`
- **Checkpoint file corruption:** Writing checkpoint in-place without tmp+rename — can leave partial JSON on crash
- **Batch job resubmission on restart:** Not saving `batch_id` to checkpoint means paying twice; always persist `batch_job_ids` list
- **Parsing stubs flowing downstream:** Returning `{"verdict": "FAIL"}` on JSONDecodeError — always return `None` and log raw response
- **Filtering out repos with empty github_url:** Must check `github_url` is non-empty AND starts with `https://github.com/` — some rows may have non-GitHub URLs

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Atomic file write | Custom lock mechanism | `tmp_path.rename(dest)` | POSIX rename is atomic; no lock file needed |
| CSV reading | Custom string split | `csv.DictReader` | Handles quoted commas in name/tags fields correctly |
| YAML generation | String templating | `yaml.safe_dump()` | Handles escaping of special chars in repo names/descriptions |
| Batch API polling | asyncio task loop | Simple `time.sleep(30)` in while-loop | No concurrency needed; 24-hour window makes polling cheap |
| JSON extraction | Growing if-elif chain | The 4-strategy `extract_json()` in utils.py | Strategies already fully enumerated; don't extend further |

**Key insight:** The `csv` module's `DictReader` correctly handles multi-value tag fields like `"drag-and-drop, editor, elementor, landing page"` (quoted in CSV) — plain `line.split(",")` breaks on these.

---

## Common Pitfalls

### Pitfall 1: CSV `active_installs` Field Contains Strings Like "10000000" (String, Not Int)

**What goes wrong:** `int(row["active_installs"]) >= 10000` raises ValueError if field contains "10000000+" or is empty.
**Why it happens:** CSV exports from WordPress.org data sometimes use "+" suffix for capped counts; confirmed in row 1 (elementor: "10000000").
**How to avoid:** Use `int(str(row.get("active_installs", 0)).replace("+", "").strip() or 0)` in filter logic.
**Warning signs:** Converter crashes on first row of plugin CSV.

### Pitfall 2: Checkpoint File Written After Every Item But Read Only at Script Start

**What goes wrong:** Script processes 500 items, writes checkpoint each time. On restart, all 500 are skipped correctly. But if the completed list grows to 5,000 items, the checkpoint JSON takes 200ms+ to read/write each iteration.
**Why it happens:** Naive `json.dump(state)` on every item with growing `completed` list.
**How to avoid:** Use a `set` for the completed list in memory; convert to list only for persistence. Or use the function-file path as the key (short string) not the full JSON payload.
**Warning signs:** Script slows down noticeably after first few hundred items.

### Pitfall 3: Batch API `custom_id` Collisions If Script Restarts

**What goes wrong:** If a batch is submitted, script crashes, and batch is NOT recorded in checkpoint, restarting the script resubmits the batch with the same `custom_id` values. Anthropic Batch API custom_id uniqueness is per-batch, not global, so this silently creates duplicate results.
**Why it happens:** `batch_job_ids` not saved to checkpoint before polling begins.
**How to avoid:** Save `batch_id` to checkpoint immediately after `submit_batch()` returns, before calling `poll_batch()`. On restart, check `batch_job_ids` and poll existing batches instead of resubmitting.
**Warning signs:** Final output has duplicate function IDs.

### Pitfall 4: `repos.yaml` Schema Mismatch Between Converter Output and `phase1_clone.py`

**What goes wrong:** The converter writes `path_filters: ["**/*.php"]` but `phase1_clone.py` and `phase1_extract.py` expect the key to be `paths:` (the existing schema uses `paths`, not `path_filters`).
**Why it happens:** CONTEXT.md uses `path_filters` terminology; existing `repos.yaml` uses `paths`/`skip_paths`.
**How to avoid:** The converter MUST output `paths:` and `skip_paths:` keys matching the existing schema (confirmed in `config/repos.yaml` lines 19-24 and 35-39). Do not invent new schema keys.
**Warning signs:** `phase1_clone.py` runs but extracts from wrong directories (all repo files).

### Pitfall 5: Pre-Flight Script Runs PHPCS Standards Check But PATH Is Wrong

**What goes wrong:** `phpcs -i` reports installed standards but "WordPress-Extra" is missing even though WPCS is installed — because PHPCS was installed globally but the Composer installer plugin didn't register the ruleset.
**Why it happens:** WPCS 3.x requires `dealerdirect/phpcodesniffer-composer-installer` to auto-register rulesets. Without it, `phpcs -i` won't list WordPress standards even if the files are present.
**How to avoid:** Pre-flight check must verify `phpcs -i` output contains "WordPress-Extra", not just that `phpcs` is in PATH. Include setup instructions for WPCS Composer install in pre-flight error message.
**Warning signs:** `phpcs --version` passes but `phpcs -i` doesn't list WordPress-Extra.

### Pitfall 6: `phase2_judge_dataset.py` Has No Rate Limiting (Confirmed in CONCERNS.md)

**What goes wrong:** Line 74 makes API call in a loop with no sleep. At 4,000 examples this fires 4,000 requests as fast as Python runs.
**Why it happens:** Rate limiting was added to all other scripts but missed here.
**How to avoid:** Add `call_with_backoff()` from utils.py as the API call mechanism (replaces direct `client.messages.create()`). The backoff handles both proactive throttling and 429 recovery.
**Warning signs:** First 429 error appears within seconds of starting `phase2_judge_dataset.py`.

### Pitfall 7: Stub Detection in Existing Output Is Fragile

**What goes wrong:** Pre-flight scans for `{"verdict": "FAIL"}` stubs but stubs from different scripts have different shapes — `phase2_judge_dataset.py` returns `None` (not a stub), while `phase1_judge.py` returns `{"verdict": "FAIL", "scores": {}}`.
**Why it happens:** Each script today has its own error return convention.
**How to avoid:** Stub detection in pre-flight should check for: JSON files where `verdict == "FAIL"` AND `scores` is empty/null, OR where required fields (`overall_score`, `critical_failures`) are missing. Use `None`-return convention uniformly after this phase.
**Warning signs:** Pre-flight reports 0 stubs even when pipeline ran with known parse failures previously.

---

## Code Examples

### CSV Column Schema (Verified)

```
# Plugins CSV columns (confirmed from file):
rank, name, slug, active_installs, rating_pct, rating_5star, editor_support,
last_updated, requires_wp, tested_up_to, homepage, github_url, wp_url,
total_known_vulns, unpatched_vulns, max_cvss, max_cvss_severity,
latest_vuln_date, top_cwes, tags

# Themes CSV columns (confirmed from file):
rank, name, slug, active_installs, rating_pct, rating_5star, editor_support,
last_updated, parent_theme, homepage, github_url, wp_url,
total_known_vulns, unpatched_vulns, max_cvss, max_cvss_severity,
latest_vuln_date, top_cwes, tags
```

### Target `repos.yaml` Schema (Must Match Exactly)

```yaml
# Confirmed from existing config/repos.yaml — use these exact keys
core:
  - name: wordpress-develop
    url: https://github.com/WordPress/wordpress-develop.git
    quality_tier: core
    paths:
      - src/wp-includes
      - src/wp-admin/includes
    skip_paths:
      - src/wp-includes/js

plugins:
  - name: yoast-seo
    url: https://github.com/Yoast/wordpress-seo.git
    quality_tier: trusted        # 0 total_known_vulns + rating >= 90
    paths:
      - src
      - includes
    skip_paths:
      - vendor
      - node_modules
      - tests
    description: "SEO plugin — 10M installs, 0 known vulns, 96% rating"

themes:
  - name: twentytwentyfive
    url: https://github.com/WordPress/twentytwentyfive.git
    quality_tier: assessed       # has known vulns or rating < 90
    paths:
      - "**/*.php"
    skip_paths:
      - vendor
      - node_modules
    description: "Official WordPress FSE theme — 1M installs"
```

Note: the `description` field is not in the current schema but is useful for traceability; confirm whether downstream scripts read it (phase1_clone.py does NOT read it — safe to add).

### Anthropic Batch API Result Parsing

```python
# Source: Anthropic Batch API docs
# Each result has: custom_id, result.type ("succeeded"/"errored"/"expired"), result.message

def parse_batch_results(results: list) -> tuple[list[dict], list[str]]:
    """Returns (successful_parsed_jsons, failed_custom_ids)."""
    successes = []
    failures = []
    for result in results:
        if result.result.type != "succeeded":
            failures.append(result.custom_id)
            continue
        text = result.result.message.content[0].text
        parsed = extract_json(text)  # from utils.py
        if parsed is None:
            failures.append(result.custom_id)
        else:
            parsed["_custom_id"] = result.custom_id
            successes.append(parsed)
    return successes, failures
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `time.sleep()` rate limiting | Exponential backoff + jitter + `retry-after` header | This phase | Survives burst 429 without crashing |
| No checkpointing | Per-file checkpoint JSON | This phase | Multi-hour scripts resumable |
| Duplicate JSON parsing in 4 scripts | Shared `utils.py` `extract_json()` | This phase | Single fix point; parse failures logged |
| Manual repos.yaml curation | CSV-driven automated conversion | This phase | ~50-100 repos from authoritative data |
| Direct API for all batch operations | Hybrid: direct (<50) / Batch API (>=50) | This phase | 50% cost reduction on bulk judging |
| PHPCS silent degradation | Fail-hard pre-flight check | This phase | No more undetectable mutations entering training data |

**Deprecated/outdated:**
- `time.sleep(REQUEST_INTERVAL)` as sole rate limiter: replaced by `call_with_backoff()` from utils.py
- `if "```json" in text: text = text.split("```json")[1].split("```")[0]` pattern: replaced by `extract_json()` in utils.py
- `except FileNotFoundError: return True` in phase2_mutate.py: replaced by hard exit

---

## Open Questions

1. **`trusted` quality_tier in phase1_clone.py/phase1_extract.py**
   - What we know: existing schema only has `core` and `assessed`; CONTEXT.md introduces `trusted`
   - What's unclear: does phase1_judge.py treat `trusted` like `core` (auto-pass) or like `assessed` (Claude judge)? CONTEXT.md says quality_tier is for CSV assignment only, but the scripts check `quality_tier == "core"` for auto-pass
   - Recommendation: add a check `if quality_tier in ("core", "trusted"): auto_pass()` in phase1_judge.py; log the addition

2. **Checkpoint directory location**
   - What we know: Claude's Discretion; options are `checkpoints/` at project root or alongside output dirs
   - What's unclear: whether `checkpoints/` should be gitignored (it should — these are runtime state files)
   - Recommendation: place at `{PROJECT_ROOT}/checkpoints/`; add to `.gitignore`

3. **Batch API polling interval**
   - What we know: Claude's Discretion; Anthropic recommends not polling more frequently than 60 seconds for large batches
   - Recommendation: use `poll_interval=60` seconds; for test/dev runs with small batches, `poll_interval=10` via CLI flag

4. **`description` field in repos.yaml**
   - What we know: current schema doesn't include it; it would be useful for traceability
   - What's unclear: whether any downstream script will break on an unexpected YAML key
   - Recommendation: add it; Python yaml.safe_load ignores unexpected keys; downstream scripts only read `name`, `url`, `quality_tier`, `paths`, `skip_paths`

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (not currently installed — Wave 0 gap) |
| Config file | none — see Wave 0 |
| Quick run command | `pytest tests/test_utils.py -x -q` |
| Full suite command | `pytest tests/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIPE-01 | Pre-flight exits with code 1 when PHPCS missing | unit | `pytest tests/test_preflight.py::test_missing_phpcs -x` | Wave 0 |
| PIPE-01 | Pre-flight exits with code 1 when API key missing | unit | `pytest tests/test_preflight.py::test_missing_api_key -x` | Wave 0 |
| PIPE-02 | Checkpoint save/load round-trips without data loss | unit | `pytest tests/test_utils.py::test_checkpoint_roundtrip -x` | Wave 0 |
| PIPE-02 | Checkpoint atomic write (tmp+rename) survives mid-write | unit | `pytest tests/test_utils.py::test_checkpoint_atomic -x` | Wave 0 |
| PIPE-03 | Backoff waits on 429, retries up to max_retries | unit (mock) | `pytest tests/test_utils.py::test_backoff_retries -x` | Wave 0 |
| PIPE-03 | Backoff reads retry-after header when present | unit (mock) | `pytest tests/test_utils.py::test_backoff_retry_after -x` | Wave 0 |
| PIPE-04 | `batch_or_direct(49)` returns "direct" | unit | `pytest tests/test_utils.py::test_routing_threshold -x` | Wave 0 |
| PIPE-04 | `batch_or_direct(50)` returns "batch" | unit | `pytest tests/test_utils.py::test_routing_threshold -x` | Wave 0 |
| PIPE-05 | `extract_json()` parses all 4 response variants correctly | unit | `pytest tests/test_utils.py::test_extract_json -x` | Wave 0 |
| PIPE-05 | `extract_json()` returns None for unparseable response | unit | `pytest tests/test_utils.py::test_extract_json_failure -x` | Wave 0 |
| REPO-01 | Converter preserves WordPress Core as first entry | unit | `pytest tests/test_csv_to_repos.py::test_core_preserved -x` | Wave 0 |
| REPO-02 | Converter produces >= 10 plugin entries meeting criteria | unit | `pytest tests/test_csv_to_repos.py::test_min_plugins -x` | Wave 0 |
| REPO-03 | Converter produces >= 5 theme entries meeting criteria | unit | `pytest tests/test_csv_to_repos.py::test_min_themes -x` | Wave 0 |
| REPO-04 | Every entry has quality_tier, paths, skip_paths | unit | `pytest tests/test_csv_to_repos.py::test_entry_schema -x` | Wave 0 |
| REPO-04 | quality_tier assigned correctly from vuln + rating data | unit | `pytest tests/test_csv_to_repos.py::test_quality_tier_logic -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_utils.py -x -q`
- **Per wave merge:** `pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/__init__.py` — package init
- [ ] `tests/test_utils.py` — covers PIPE-02, PIPE-03, PIPE-04, PIPE-05
- [ ] `tests/test_preflight.py` — covers PIPE-01
- [ ] `tests/test_csv_to_repos.py` — covers REPO-01 through REPO-04
- [ ] `tests/fixtures/sample_plugins.csv` — 5-row sample for CSV tests
- [ ] `tests/fixtures/sample_themes.csv` — 5-row sample for CSV tests
- [ ] Framework install: `pip install pytest` — not currently present

---

## Sources

### Primary (HIGH confidence)
- Direct codebase audit: `scripts/phase1_judge.py`, `scripts/phase2_judge_dataset.py`, `scripts/phase1_clone.py`, `scripts/phase2_generate.py` — JSON parsing patterns, rate limiting gaps confirmed at specific line numbers
- `.planning/codebase/CONCERNS.md` — All issues verified against actual source files
- `config/repos.yaml` — Exact schema confirmed (keys: name, url, quality_tier, paths, skip_paths, notes)
- `/home/robert_li/Desktop/data/wp-finetune-data/wp_top1000_plugins_final.csv` — Header row confirmed; sample rows verified
- `/home/robert_li/Desktop/data/wp-finetune-data/wp_top100_themes_final.csv` — Header row confirmed; Astra row shows empty github_url (filtering needed)

### Secondary (MEDIUM confidence)
- [Anthropic Batch API docs](https://docs.anthropic.com/en/api/creating-message-batches) — `client.beta.messages.batches` API, request format, result structure
- [Anthropic Rate Limits](https://platform.claude.com/docs/en/api/rate-limits) — RPM/TPM limits, retry-after header behavior
- `.planning/research/PITFALLS.md` — Pitfall 1 (parse failures), Pitfall 2 (rate limits), Pitfall 3 (checkpointing), Pitfall 4 (PHPCS degradation) — all grounded in codebase audit

### Tertiary (LOW confidence)
- tenacity library: not installed in local env; mentioned as Claude's Discretion option only

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use or stdlib; no new deps required
- Architecture: HIGH — patterns derived from existing code + locked decisions in CONTEXT.md
- CSV schema: HIGH — verified by reading actual file headers and sample rows
- Batch API format: MEDIUM — derived from official docs; not tested against live API in this project yet
- Pitfalls: HIGH — 6 of 7 pitfalls confirmed by direct code inspection

**Research date:** 2026-03-26
**Valid until:** 2026-04-26 (Anthropic Batch API beta namespace may change; re-verify if SDK updated)
