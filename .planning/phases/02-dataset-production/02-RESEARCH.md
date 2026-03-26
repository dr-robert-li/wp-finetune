# Phase 2: Dataset Production - Research

**Researched:** 2026-03-26
**Domain:** Multi-stage LLM data pipeline — PHP extraction, Claude judging, synthetic generation, dataset assembly
**Confidence:** HIGH (direct codebase audit of all 10 pipeline scripts + config files)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Claude judge threshold raised to >= 8** (from >= 7) — higher quality bar, fewer but better examples
- **PHPCS pre-filter stays at < 5 errors/100 lines** — unchanged
- **Trusted repos still go through full assessment** — even 0-vuln repos may have individual bad functions
- **Synthetic revision: 1 retry** (current behavior) — failed synthetics get one revision attempt, then discarded
- **Aggressive critical_failures:** Any single security dimension score < 5 = automatic FAIL regardless of overall score. Update `config/judge_system.md` before execution.
- **40/60 gen/judge split** instead of 50/50 — emphasize critic capability. Update `export_dataset.py` ratio.
- **Rejection examples in training data:** ~500 examples where model proactively adds security measures. Generate during Phase 2 synthetic generation.
- **Contrastive/low-score examples weighted higher** — include more bad→good pairs with CoT explanations
- **If >50% rejection rate on extracted code:** Pull additional repos from remaining ~950 plugins in CSV data
- **If <10,000 examples after full pipeline:** Add more repos first, then increase synthetic generation targets
- **Taxonomy categories with <20 examples:** Flag but don't block
- **Automated stats + spot check:** ~20 random examples reviewed by Claude Code for teaching quality
- **Report:** Generate `final_dataset/metadata.json` with full stats before declaring Phase 2 done

### Claude's Discretion
- How to wire `scripts/utils.py` functions into existing 8 pipeline scripts
- Whether to refactor in-place or add wrapper layer
- Batch API batch sizes and polling intervals
- Order of execution within pipeline phases
- Exact execution sequence and parallelism within pipeline phases
- How to handle partial Batch API failures
- Synthetic prompt template adjustments for rejection examples
- Taxonomy minimum thresholds per category

### Deferred Ideas (OUT OF SCOPE)
- Adversarial examples from Phase D4 fed back into training data — belongs in v2 training cycle
- The dual-mode architecture insight is a training-time concern for Phase 3 config
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Phase 1 clone completes — all repos in repos.yaml shallow-cloned | `phase1_clone.py` needs checkpoint integration via `utils.load_checkpoint` / `utils.save_checkpoint`; 56 repos in `config/repos.yaml` ready |
| DATA-02 | Phase 1 extract completes — PHP functions extracted with metadata | `phase1_extract.py` needs checkpoint integration; `php_extract_functions.php` already exists; extraction happens per-repo |
| DATA-03 | Phase 1 judge completes — functions assessed (PHPCS pre-filter + Claude judge), passed/failed separated | `phase1_judge.py` needs: `extract_json` replacing brittle split parser, `call_with_backoff` replacing `time.sleep`, checkpoint integration, Batch API routing; judge threshold update to >= 8 in `config/judge_system.md`; security dim < 5 = auto-FAIL |
| DATA-04 | Phase 2 gap analysis completes — coverage gaps identified against taxonomy | `phase2_gap_analysis.py` is already clean (no API calls); just needs to run after DATA-03 |
| DATA-05 | Phase 2 mutation completes — contrastive bad→good pairs generated from passed code | `phase2_mutate.py` needs PHPCS hard-fail guard (currently silently accepts mutations when PHPCS absent); no API integration needed |
| DATA-06 | Phase 2 generate completes — synthetic examples fill taxonomy gaps + ~500 rejection examples | `phase2_generate.py` needs: `call_with_backoff`, checkpoint integration, Batch API routing, new rejection example templates in `config/synthetic_prompts.yaml` |
| DATA-07 | Phase 2 judge completes — synthetic examples assessed, failed get one revision | `phase2_judge.py` needs: `extract_json`, `call_with_backoff`, checkpoint integration |
| DATA-08 | Phase 2 judge_dataset completes — rubric-scored judge training data generated | `phase2_judge_dataset.py` needs: rate limiting (confirmed missing in CONCERNS.md), `extract_json`, `call_with_backoff`, Batch API routing |
| DATA-09 | Phase 3 CoT completes — instruction synthesis + reasoning chains generated | `phase3_cot.py` needs: `call_with_backoff`, checkpoint integration (already has rudimentary checkpointing at 500 examples but uses local file writes not utils.py) |
| DATA-10 | Phase 3 export completes — OpenAI, Alpaca, Raw JSONL with task tokens, 80/10/10 split; 40/60 gen/judge ratio enforcement | `export_dataset.py` needs: ratio enforcement for 40/60 gen/judge split, dataset validation, PHP validity check via `php -l`, deduplication across train/val splits |
| DATA-11 | Final dataset contains >= 10,000 examples with ~40/60 wp_gen/wp_judge split | Requires fallback strategy if extraction yield is low; metadata.json generation; spot-check validation |
</phase_requirements>

---

## Summary

Phase 2 is an execution phase, not a design phase. All 10 pipeline scripts already exist. The work is: (1) wire `scripts/utils.py` into 8 scripts by replacing brittle patterns, (2) update two config files (`judge_system.md`, `synthetic_prompts.yaml`, `export_dataset.py`), and (3) run the pipeline end-to-end on 56 repos.

The hardening integration is strictly surgical: replace `json.loads(text.split("```json")[1]...)` with `extract_json()`, replace `time.sleep(REQUEST_INTERVAL)` with `call_with_backoff()`, add `load_checkpoint`/`save_checkpoint` at script boundaries, and add `batch_or_direct()` routing for scripts processing >= 50 items. No new pipeline architecture is introduced.

The three config changes are discrete: raise the PASS threshold in `judge_system.md` from `>= 7` to `>= 8`, add security-dimension auto-FAIL rule (`< 5` on any security dimension = FAIL), add rejection example prompt templates to `synthetic_prompts.yaml`, and update the gen/judge ratio constant in `export_dataset.py` from implied 50/50 to 40/60.

**Primary recommendation:** Integrate utils.py into scripts in dependency order (scripts earlier in the pipeline first), make config changes before any execution, then run the pipeline in full with `--sample` testing before committing to the full run. Address the PHPCS hard-fail guard in `phase2_mutate.py` before running DATA-05.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| anthropic (Python SDK) | already installed | Claude API calls, Batch API | All scripts already use `anthropic.Anthropic()` |
| PyYAML | already installed | Read `config/repos.yaml`, `taxonomy.yaml`, `synthetic_prompts.yaml` | Used across all pipeline scripts |
| pytest | used in Phase 1 tests | Unit tests for utils integration | 15 tests already passing in `tests/test_utils.py` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| subprocess | stdlib | PHPCS, php-lint, git clone | Used in phase1_clone, phase1_extract, phase1_judge, phase2_mutate |
| pathlib | stdlib | All file I/O paths | Consistent path handling across all scripts |
| json | stdlib | Checkpoint, output, config | All scripts |
| re | stdlib | Mutation detection patterns | phase2_mutate.py |

### Not Needed
- No new libraries. Phase 2 adds no new dependencies — it wires existing utils.py into existing scripts.

---

## Architecture Patterns

### Existing Pipeline Structure (DO NOT CHANGE)
```
phase1_extraction/
├── repos/           # Cloned repos (DATA-01)
└── output/
    ├── extracted/   # Per-repo JSON (DATA-02)
    ├── passed/      # Judge-approved functions (DATA-03)
    └── failed/      # Rejected functions (DATA-03)

phase2_synthetic/
├── gap_report.json              # (DATA-04)
└── output/
    ├── mutated/                 # Contrastive pairs (DATA-05)
    ├── generated/               # Synthetic fills + rejection examples (DATA-06)
    ├── judged/                  # Assessed synthetics (DATA-07)
    └── judge_training/          # Scored judge examples (DATA-08)

phase3_cot/
└── output/                      # CoT checkpoint files (DATA-09)

final_dataset/
├── wordpress_finetune.jsonl     # Source for export (DATA-09 output)
├── metadata.json                # Stats report (DATA-11)
├── openai_{train,val,test}.jsonl
├── alpaca_{train,val,test}.json
└── raw_{train,val,test}.jsonl   # (DATA-10)

checkpoints/                     # utils.py checkpoint files
```

### Pattern 1: utils.py Integration — In-Place Replacement
**What:** Replace brittle code directly at call sites in each script. No wrapper layer.
**When to use:** All 8 scripts that need hardening.
**Integration points per script:**

#### phase1_clone.py
- Add `from scripts.utils import load_checkpoint, save_checkpoint` at top
- Load checkpoint at start of `main()`, skip already-cloned repos, save checkpoint per-repo
- No API calls — no backoff needed

#### phase1_extract.py
- Add checkpoint integration: skip already-extracted repos
- No API calls — no backoff needed

#### phase1_judge.py (largest integration, DATA-03)
- Replace `time.sleep(REQUEST_INTERVAL)` with `call_with_backoff()`
- Replace brittle JSON parsing in `judge_function()` (lines 195-200) with `extract_json()`
- Add `load_checkpoint`/`save_checkpoint` wrapping the repo loop
- Add `batch_or_direct()` routing: if functions_to_judge >= 50, use Batch API path
- Remove `REQUESTS_PER_MINUTE` and `REQUEST_INTERVAL` constants (replaced by backoff)

#### phase2_gap_analysis.py
- No API calls, no JSON parsing from API — NO integration needed
- Just run it as-is after DATA-03 completes

#### phase2_mutate.py
- No API calls — NO backoff/extract_json integration
- ADD: hard-fail guard if PHPCS unavailable (replace `return True` in `verify_mutation_detectable()`)
- ADD: call `run_preflight()` or inline PHPCS check at script start

#### phase2_generate.py
- Replace `time.sleep(REQUEST_INTERVAL)` with `call_with_backoff()` in `generate_one()`
- Add checkpoint integration: skip already-generated gap_tags on resume
- Add `batch_or_direct()` routing: if deficit >= 50, submit as Batch API job
- ADD new rejection example section (see Pattern 2 below)

#### phase2_judge.py
- Replace brittle JSON parsing in `judge_synthetic()` with `extract_json()`
- Replace `time.sleep(REQUEST_INTERVAL)` with `call_with_backoff()`
- Add checkpoint integration per gen_file
- Add Batch API routing: if examples >= 50, batch judge them

#### phase2_judge_dataset.py
- Replace brittle JSON parsing in `generate_judge_score()` with `extract_json()`
- ADD `call_with_backoff()` (confirmed missing — PIPE-03 fix, CONCERNS.md line 16-17)
- Add checkpoint integration: save every 100 scored examples
- Add Batch API routing: samples >= 50 → batch mode

#### phase3_cot.py
- Replace `client.messages.create(...)` calls with `call_with_backoff()`
- Replace existing rudimentary checkpointing (lines 305-312) with `save_checkpoint()`/`load_checkpoint()`
- No JSON parsing from API (CoT returns text, not JSON)

#### export_dataset.py
- Update gen/judge ratio: add `GEN_RATIO = 0.40` / `JUDGE_RATIO = 0.60` constants
- Add ratio enforcement in `main()`: sample from gen/judge pools to hit 40/60
- Add dataset validation: PHP lint via `php -l`, duplicate detection, task token presence
- Add `final_dataset/metadata.json` generation with full stats

### Pattern 2: Rejection Example Templates
**What:** New synthetic generation category where model proactively adds security measures.
**When to use:** Phase 2 generation (DATA-06), ~500 examples targeting `rejection:proactive_security`.
**How to add:** New section in `config/synthetic_prompts.yaml` under a new key `rejection_templates`.

Example prompt pattern:
```yaml
rejection_templates:
  proactive_nonce:
    - >
      Write a WordPress form handler for a {context} that processes user-submitted data
      and saves settings. The prompt from the user simply says "handle form submission"
      without mentioning security. Your response MUST proactively add nonce verification
      and explain CSRF risk in a code comment, even though the user did not ask for it.
  proactive_capability:
    - >
      Write a WordPress admin page action handler. The task description only says "process
      the admin form". Your response MUST proactively add current_user_can() checks and
      explain privilege escalation risk in comments, even though the user did not ask for it.
  proactive_escaping:
    - >
      Write a WordPress template function that displays user-submitted content in {context}.
      The requirement only says "display the content". Your response MUST proactively apply
      esc_html() or wp_kses() and explain XSS risk in comments, even though not requested.
```

These get a new training tag `rejection:proactive_{type}` and are treated as `wp_gen` examples (the model generates code AND security explanation).

### Pattern 3: Judge Threshold Update
**What:** Update `config/judge_system.md` to reflect >= 8 threshold and security dimension auto-FAIL.
**Where:** Lines 17-18 of current `judge_system.md`:
- Change: `A PASS requires ALL dimensions >= 7 and no critical failures.`
- To: `A PASS requires ALL dimensions >= 8 and no critical failures.`
- Add: `SECURITY AUTO-FAIL: If the security dimension (dimension 3) scores < 5, the verdict is automatically FAIL regardless of all other scores.`

This must happen BEFORE running DATA-03. All existing assessed functions in `phase1_extraction/output/` must be re-judged with the new threshold (they don't exist yet — pipeline hasn't run).

### Pattern 4: Batch API Integration
**What:** Use `make_batch_request()`, `submit_batch()`, `poll_batch()`, `parse_batch_results()` for offline bulk processing.
**When to use:** When item count >= 50 (BATCH_THRESHOLD constant).
**Applicable scripts:** phase1_judge.py (all assessed functions), phase2_generate.py (large gaps), phase2_judge.py (judging generated), phase2_judge_dataset.py (scoring samples).
**Critical:** Batch API is async — `poll_batch()` sleeps 60s between polls. Scripts using Batch API must handle the wait and store `batch_id` in checkpoint so they can resume polling after crashes.

Checkpoint structure for batch-mode scripts:
```python
state = {
    "completed": [],         # repo/item names fully processed
    "failed": [],            # repo/item names that permanently failed
    "batch_job_ids": [],     # pending batch IDs for resume
    "timestamp": None,
}
```

### Anti-Patterns to Avoid
- **Don't use `time.sleep(REQUEST_INTERVAL)` as the primary rate limiter:** Replace with `call_with_backoff()`. The sleep doesn't handle 429 or 5xx errors — it just adds fixed latency.
- **Don't use brittle string splitting for JSON extraction:** `text.split("```json")[1].split("```")[0]` breaks on any variation. Replace with `extract_json()`.
- **Don't add a wrapper layer around existing scripts:** Modify in-place. A wrapper adds indirection without benefit for scripts that are already isolated.
- **Don't run phase2_mutate.py without verifying PHPCS is active:** The silent `return True` fallback corrupts training data. Fail hard.
- **Don't reuse code examples across gen and judge training sets:** Leads to train/val leakage (Pitfall 9 in PITFALLS.md). Track by code hash.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON extraction from LLM responses | Custom regex parser | `extract_json()` in utils.py | 4-strategy fallback already tested, handles all common Claude response formats |
| API retry on 429/5xx | Custom `while True` loop | `call_with_backoff()` in utils.py | Handles `retry_after` header, jitter, max_retries, both RateLimitError and APIStatusError |
| Checkpoint save/load | Custom JSON write | `save_checkpoint()`/`load_checkpoint()` in utils.py | Atomic rename, correct schema, already tested in `tests/test_utils.py` |
| Batch/direct routing decision | `if count > 50` inline | `batch_or_direct()` in utils.py | Consistent threshold, BATCH_THRESHOLD constant already aligned with PIPE-04 spec |
| Pre-flight validation | Inline subprocess checks | `run_preflight()` from preflight.py | Already checks ANTHROPIC_API_KEY, PHP CLI, PHPCS, WordPress-Extra standard |
| PHP code lint check | Custom PHP parser | `php -l /path/to/file.php` via subprocess | Built-in PHP syntax checker, exit code 0 = valid, non-zero = syntax error |

**Key insight:** utils.py is the Phase 1 deliverable specifically designed to be wired into Phase 2. Every "don't hand-roll" item in Phase 2 was already implemented and tested in Phase 1. The integration is the work.

---

## Common Pitfalls

### Pitfall 1: Judge Threshold Change Not Propagated Before Execution
**What goes wrong:** Running DATA-03 with the old `>= 7` threshold, then realizing the change was not made. Must re-run entire Phase 1 judge on all 56 repos.
**Why it happens:** `config/judge_system.md` is read at runtime by `load_judge_system()`. If the file is not updated first, the old threshold silently applies.
**How to avoid:** Make the `judge_system.md` and `synthetic_prompts.yaml` config changes in Wave 0 (a dedicated setup task) BEFORE any execution task runs.
**Warning signs:** Pass rates higher than expected after raising threshold.

### Pitfall 2: phase2_mutate.py Silently Accepts All Mutations Without PHPCS
**What goes wrong:** `verify_mutation_detectable()` returns `True` when PHPCS raises `FileNotFoundError`. Undetectable mutations enter judge training data. Model trains on invisible defects.
**Why it happens:** Graceful degradation chosen over fail-fast (confirmed in CONCERNS.md lines 91-103).
**How to avoid:** Replace the `except FileNotFoundError: return True` with an explicit PHPCS availability check at script start using `run_preflight()` or a targeted subprocess check. Exit with error if PHPCS missing.
**Warning signs:** Mutation acceptance rate is 100%; script completes instantly without PHPCS output.

### Pitfall 3: parse_batch_results() Returning Only Partial Results After Crash
**What goes wrong:** A batch job completes, `poll_batch()` retrieves results, but the script crashes before saving. On resume, the batch_id is still in the checkpoint but the Batch API may no longer return results (results expire after 24 hours per Anthropic docs).
**Why it happens:** Checkpoint saves `batch_job_ids` but doesn't distinguish between "submitted, awaiting", "polled, results saved", and "polled, failed to save".
**How to avoid:** Save batch results to a local file immediately after `parse_batch_results()` returns, before updating the checkpoint. Use the pattern: save results → update checkpoint → continue. Structure: `checkpoints/{phase}_batch_{batch_id}_results.json`.

### Pitfall 4: 40/60 Ratio Enforcement Creates Insufficient Gen Examples
**What goes wrong:** If the pipeline produces, say, 4,000 gen examples and 8,000 judge examples, a strict 40/60 ratio means only 4,000 gen examples and 6,000 judge examples (total 10,000). But if the pipeline produces 2,000 gen and 8,000 judge, strict 40/60 would need 2,000 gen + 3,000 judge = only 5,000 total — below the 10,000 minimum.
**Why it happens:** 40/60 is a target ratio, not a floor guarantee. If gen yield is low, ratio enforcement reduces total count.
**How to avoid:** In `export_dataset.py`: enforce ratio as a cap on the majority class, not as a minimum. If gen_count < 40% of total, use all gen examples and cap judge examples at `gen_count * (60/40)`. Report actual ratio in metadata.json alongside the target.

### Pitfall 5: Phase 3 CoT Checkpoint Using File Writes Instead of utils.py
**What goes wrong:** `phase3_cot.py` (lines 305-312) already has checkpoint-like behavior writing to `phase3_cot/output/checkpoint_{N}.jsonl`. This uses a different format and location from utils.py checkpoints. If both systems coexist, resume logic becomes confused.
**Why it happens:** Phase 3 was written before utils.py existed.
**How to avoid:** Replace the existing Phase 3 checkpoint write with `save_checkpoint("phase3_cot", state)`. Migrate the local checkpoint file format to the utils.py schema. Keep the per-500 intermediate saves as progress files (useful for recovery) but use utils.py as the authoritative resume pointer.

### Pitfall 6: Security Dimension N/A Inflation Still Active After Threshold Change
**What goes wrong:** Backend functions that produce no HTML receive score 10 for accessibility and i18n. With the new threshold at >= 8, these inflated N/A scores push functions over the bar that shouldn't pass on security/WPCS merits alone.
**Why it happens:** N/A handling in `judge_system.md` is unchanged (CONCERNS.md line 125-143). The threshold change makes this worse, not better, since the inflated scores now help functions cross a higher bar.
**How to avoid:** When updating `judge_system.md` for the threshold change, also update the N/A handling: change "Score N/A (10)" to "Score N/A (7)" — or better, add a note that N/A dimensions are excluded from average calculation. The planner should flag this as a required config change alongside the threshold update.

### Pitfall 7: Rejection Examples Getting Mislabeled as wp_judge
**What goes wrong:** Rejection examples have the model producing code + security explanation. `infer_task_type()` in `export_dataset.py` looks for `<wp_judge>` in the user message. If the rejection example template doesn't explicitly include `<wp_gen>`, it defaults to "gen" correctly — but if a rejection example mentions "evaluate" or "review," it might be miscategorized as judge.
**Why it happens:** The `infer_task_type()` default is "gen" but the detection logic searches for judge-indicating keywords.
**How to avoid:** Rejection example templates should explicitly frame the user message as a generation task ("Write a WordPress function that...") and set `metadata.task_type = "gen"` explicitly in the generated output.

---

## Code Examples

### Extract JSON Integration (replaces brittle split pattern)
```python
# BEFORE (all 3 judge scripts):
text = response.content[0].text
if "```json" in text:
    text = text.split("```json")[1].split("```")[0]
elif "```" in text:
    text = text.split("```")[1].split("```")[0]
return json.loads(text.strip())

# AFTER (using utils.py):
from scripts.utils import extract_json
text = response.content[0].text
result = extract_json(text)
if result is None:
    # Log and hard-reject, never substitute stub
    print(f"  PARSE_FAIL: {func['function_name']}", file=sys.stderr)
    return None  # Caller skips None result
```

### call_with_backoff Integration (replaces time.sleep pattern)
```python
# BEFORE:
response = client.messages.create(
    model="claude-sonnet-4-6-20250514",
    max_tokens=1024,
    system=system,
    messages=[{"role": "user", "content": prompt}],
)
time.sleep(REQUEST_INTERVAL)

# AFTER:
from scripts.utils import call_with_backoff
response = call_with_backoff(
    client,
    model="claude-sonnet-4-6-20250514",
    max_tokens=1024,
    system=system,
    messages=[{"role": "user", "content": prompt}],
)
# No sleep needed — backoff handles rate limits reactively
```

### Checkpoint Integration Pattern (for repo-loop scripts)
```python
from scripts.utils import load_checkpoint, save_checkpoint

def main():
    checkpoint = load_checkpoint("phase1_judge")
    completed = set(checkpoint["completed"])

    for repo_file in extracted_files:
        repo_name = repo_file.stem
        if repo_name in completed:
            print(f"  [{repo_name}] Skipping (checkpointed)")
            continue

        # ... process repo ...

        completed.add(repo_name)
        save_checkpoint("phase1_judge", {
            "completed": list(completed),
            "failed": checkpoint["failed"],
            "batch_job_ids": checkpoint["batch_job_ids"],
        })
```

### Batch API Integration Pattern (for scripts with >= 50 items)
```python
from scripts.utils import (
    batch_or_direct, make_batch_request, submit_batch, poll_batch, parse_batch_results
)

def process_with_batch_or_direct(items, client, system):
    mode = batch_or_direct(len(items))
    if mode == "batch":
        requests = [
            make_batch_request(
                custom_id=f"item-{i}",
                system=system,
                user_content=build_prompt(item),
                model="claude-sonnet-4-6-20250514",
                max_tokens=1024,
            )
            for i, item in enumerate(items)
        ]
        batch_id = submit_batch(client, requests)
        results_raw = poll_batch(client, batch_id, poll_interval=60)
        successes, failures = parse_batch_results(results_raw)
        return successes, failures
    else:
        # Direct processing via call_with_backoff
        ...
```

### PHPCS Hard-Fail Guard for phase2_mutate.py
```python
# Add at start of main() in phase2_mutate.py:
import subprocess

def _require_phpcs():
    """Exit hard if PHPCS is not available."""
    try:
        result = subprocess.run(
            ["phpcs", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print(
            "ERROR: phpcs not found. Mutation detection requires PHPCS. "
            "Install via: composer global require squizlabs/php_codesniffer",
            file=sys.stderr,
        )
        sys.exit(1)

def main():
    _require_phpcs()
    # ... rest of main
```

### 40/60 Ratio Enforcement in export_dataset.py
```python
# Replace current shuffle-and-split with ratio-aware split:
GEN_TARGET_RATIO = 0.40
JUDGE_TARGET_RATIO = 0.60

def enforce_ratio(dataset: list[dict]) -> list[dict]:
    """Enforce 40/60 gen/judge ratio by capping the majority class."""
    gen_examples = [ex for ex in dataset if infer_task_type(ex) == "gen"]
    judge_examples = [ex for ex in dataset if infer_task_type(ex) == "judge"]

    gen_count = len(gen_examples)
    judge_count = len(judge_examples)

    # Cap the majority class to achieve target ratio
    if gen_count > 0 and judge_count > 0:
        ideal_judge = int(gen_count * (JUDGE_TARGET_RATIO / GEN_TARGET_RATIO))
        if judge_count > ideal_judge:
            judge_examples = random.sample(judge_examples, ideal_judge)
        else:
            # Judge is the limiting factor; gen is capped instead
            ideal_gen = int(judge_count * (GEN_TARGET_RATIO / JUDGE_TARGET_RATIO))
            if gen_count > ideal_gen:
                gen_examples = random.sample(gen_examples, ideal_gen)

    return gen_examples + judge_examples
```

### Metadata.json with Full Stats (for DATA-11)
The existing `phase3_cot.py` already writes `final_dataset/metadata.json` but lacks the user-required fields. Extend it:
```python
metadata = {
    # Existing fields:
    "total_examples": len(training_data),
    "cot_examples": cot_count,
    "direct_examples": direct_count,
    "judge_examples": judge_count,
    # New required fields:
    "gen_judge_ratio": f"{gen_count}/{judge_count}",
    "gen_ratio_actual": round(gen_count / len(training_data), 3),
    "judge_ratio_actual": round(judge_count / len(training_data), 3),
    "rejection_examples": sum(1 for td in training_data
        if "rejection:" in str(td.get("metadata", {}).get("tags", []))),
    "taxonomy_coverage": {tag: count for tag, count in tag_counts.items()},
    "taxonomy_gaps_remaining": [tag for tag, min_count in minimums.items()
        if tag_counts.get(tag, 0) < min_count],
    "train_val_test_counts": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
    "php_lint_failures": php_lint_failure_count,
    "spot_check_required": True,
    "phase2_complete": False,  # Set to True after spot-check passes
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `time.sleep(REQUEST_INTERVAL)` fixed rate limiter | `call_with_backoff()` with exponential backoff + retry-after | Phase 1 deliverable | Eliminates crashes on 429; handles Anthropic's retry-after header |
| Brittle `split("```json")` parsing | `extract_json()` 4-strategy extraction | Phase 1 deliverable | Handles all Claude response format variations without stubs |
| No checkpointing | `load_checkpoint`/`save_checkpoint` with atomic rename | Phase 1 deliverable | Scripts survive crashes and resume without data loss |
| Realtime API for all volumes | `batch_or_direct()` routing + Batch API for >= 50 items | Phase 1 deliverable | 50% cost reduction, eliminates RPM pressure for bulk operations |
| Judge threshold >= 7 | Judge threshold >= 8 | Phase 2 decision | Higher quality floor; fewer but better training examples |
| No rejection examples | ~500 proactive security examples | Phase 2 decision | Model learns to add security without being asked |
| 50/50 gen/judge split | 40/60 gen/judge split | Phase 2 decision | Stronger critic capability emphasis |

---

## Open Questions

1. **Rejection examples word count target**
   - What we know: ~500 rejection examples requested
   - What's unclear: Whether 500 is the final target for the judged dataset (pre-rejection-judge) or post-judged. Given the 1-retry pattern, if 30% fail after retry, we need ~715 generation attempts to yield 500.
   - Recommendation: Generate 700 rejection examples in phase2_generate.py (targeting 500 survivors after phase2_judge.py).

2. **Core code in style anchors**
   - What we know: `phase2_generate.py` uses Phase 1 passed code as style anchors. WordPress Core is auto-passed (no Claude judgment). CONCERNS.md notes Core code may have subtle SQL vulnerabilities that auto-tagging doesn't catch.
   - What's unclear: Whether Core-sourced style anchors should be excluded or kept.
   - Recommendation: Mark Core-sourced anchors as "reference style only, not security validated" in the prompt. Don't exclude them — Core's naming conventions and hook patterns are valuable style references.

3. **Contrastive examples weighting in export**
   - What we know: CONTEXT.md says "contrastive/low-score examples weighted higher." `export_dataset.py` does random sampling.
   - What's unclear: How to implement "weighting" — duplicate examples, bias sampling, or metadata flag for training.
   - Recommendation: Add a `sample_weight` field to metadata for contrastive/low-score examples. Since downstream training (Phase 3) controls loss weighting, the export just needs to preserve the signal; no duplication in the JSONL.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (confirmed in Phase 1 tests) |
| Config file | none — invoked via `pytest tests/` from project root |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Checkpoint skips already-cloned repos | unit | `pytest tests/test_pipeline_integration.py::test_clone_checkpoint_skip -x` | Wave 0 |
| DATA-02 | Checkpoint skips already-extracted repos | unit | `pytest tests/test_pipeline_integration.py::test_extract_checkpoint_skip -x` | Wave 0 |
| DATA-03 | `extract_json()` replaces brittle split in judge | unit | `pytest tests/test_utils.py -x` (existing) | YES |
| DATA-03 | `call_with_backoff()` replaces sleep in judge | unit | `pytest tests/test_utils.py::test_backoff_retries -x` (existing) | YES |
| DATA-03 | Judge threshold >= 8 in system prompt | unit | `pytest tests/test_config.py::test_judge_threshold_v2 -x` | Wave 0 |
| DATA-05 | PHPCS hard-fail guard in phase2_mutate | unit | `pytest tests/test_phase2_mutate.py::test_phpcs_required -x` | Wave 0 |
| DATA-06 | Rejection example templates exist in synthetic_prompts.yaml | unit | `pytest tests/test_config.py::test_rejection_templates_exist -x` | Wave 0 |
| DATA-08 | `call_with_backoff()` in phase2_judge_dataset | unit | `pytest tests/test_phase2_judge_dataset.py::test_rate_limiting -x` | Wave 0 |
| DATA-10 | 40/60 gen/judge ratio enforced in export | unit | `pytest tests/test_export.py::test_gen_judge_ratio -x` | Wave 0 |
| DATA-10 | PHP lint validation in export | unit | `pytest tests/test_export.py::test_php_lint_validation -x` | Wave 0 |
| DATA-11 | metadata.json contains required fields | unit | `pytest tests/test_export.py::test_metadata_fields -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_pipeline_integration.py` — covers DATA-01, DATA-02 (checkpoint skip behavior)
- [ ] `tests/test_config.py` — covers DATA-03 threshold, DATA-06 rejection templates
- [ ] `tests/test_phase2_mutate.py` — covers DATA-05 PHPCS hard-fail guard
- [ ] `tests/test_phase2_judge_dataset.py` — covers DATA-08 rate limiting fix
- [ ] `tests/test_export.py` — covers DATA-10 ratio enforcement, PHP lint, DATA-11 metadata fields

*(Existing `tests/test_utils.py` already covers `extract_json`, `call_with_backoff`, `load_checkpoint`/`save_checkpoint`, `batch_or_direct` — 15 passing tests. No gaps for utils.py itself.)*

---

## Sources

### Primary (HIGH confidence)
- Direct codebase audit: all 10 pipeline scripts read in full
  - `scripts/phase1_clone.py`, `phase1_extract.py`, `phase1_judge.py`
  - `scripts/phase2_gap_analysis.py`, `phase2_mutate.py`, `phase2_generate.py`, `phase2_judge.py`, `phase2_judge_dataset.py`
  - `scripts/phase3_cot.py`, `export_dataset.py`
  - `scripts/utils.py` (334 lines, 9 functions), `scripts/preflight.py` (85 lines)
- `config/judge_system.md` — current judge criteria (threshold, dimensions, N/A handling)
- `config/taxonomy.yaml` — 27 concept tags with minimum_coverage targets
- `config/synthetic_prompts.yaml` — existing generation templates
- `config/repos.yaml` — 56 repos confirmed ready for execution
- `.planning/codebase/CONCERNS.md` — full codebase issues audit (2026-03-26)
- `.planning/research/PITFALLS.md` — 12 pitfalls with prevention strategies (2026-03-26)
- `.planning/phases/02-dataset-production/02-CONTEXT.md` — locked user decisions
- `.planning/REQUIREMENTS.md` — DATA-01 through DATA-11 definitions
- `tests/test_utils.py` — 15 existing passing tests confirming utils.py API

### Secondary (MEDIUM confidence)
- `PROJECT.md` — pipeline composition targets, dataset size goals
- `.planning/STATE.md` — confirmed Phase 1 complete with 9 utils functions + 15 tests passing

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries; all tools confirmed present in existing scripts
- Architecture: HIGH — direct code audit of all integration points
- Pitfalls: HIGH — grounded in CONCERNS.md codebase analysis + PITFALLS.md research

**Research date:** 2026-03-26
**Valid until:** 2026-04-26 (stable domain — only changes if scripts are modified before planning starts)
