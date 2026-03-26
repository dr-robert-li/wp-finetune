# Phase 2: Dataset Production - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Execute the full 3-phase data pipeline against real repositories to produce a clean, split, multi-format training dataset. This integrates Phase 1 hardening (utils.py) into the 8 existing pipeline scripts and runs them end-to-end. No new pipeline features — execution of existing scripts with hardening wired in.

</domain>

<decisions>
## Implementation Decisions

### Quality Thresholds
- **Claude judge threshold raised to >= 8** (from >= 7) — higher quality bar, fewer but better examples
- **PHPCS pre-filter stays at < 5 errors/100 lines** — unchanged
- **Trusted repos still go through full assessment** — even 0-vuln repos may have individual bad functions
- **Synthetic revision: 1 retry** (current behavior) — failed synthetics get one revision attempt, then discarded
- **Aggressive critical_failures:** Any single security dimension score < 5 = automatic FAIL regardless of overall score. Update `config/judge_system.md` before execution.

### Dataset Composition — Push Back Harder
- **40/60 gen/judge split** instead of 50/50 — emphasize critic capability. Update `export_dataset.py` ratio.
- **Rejection examples in training data:** Add prompts where the model should proactively add security measures even when the prompt doesn't mention them. E.g., "write a form handler" without mentioning nonces → model responds with nonce verification and explains why. Generate ~500 rejection examples during Phase 2 synthetic generation.
- **Contrastive/low-score examples weighted higher** — include more bad→good pairs with CoT explanations of what's wrong and why

### Fallback Strategy
- **If >50% rejection rate on extracted code:** Pull additional repos from remaining ~950 plugins in CSV data (re-run csv_to_repos.py with relaxed filters or larger cap)
- **If <10,000 examples after full pipeline:** Add more repos first, then increase synthetic generation targets
- **Taxonomy categories with <20 examples:** Flag but don't block — rare categories (multisite, cron) may need extra synthetic generation

### Dataset Validation
- **Automated stats:** Example count, split ratios, task token presence, taxonomy coverage report, gen/judge balance verification
- **Spot check:** Claude Code reviews ~20 random examples for correctness and teaching quality (would this teach the model the right thing?)
- **Report:** Generate `final_dataset/metadata.json` with full stats before declaring Phase 2 done

### Script Integration (Claude's Discretion)
- How to wire `scripts/utils.py` functions (extract_json, call_with_backoff, save/load_checkpoint, batch_or_direct) into existing 8 pipeline scripts
- Whether to refactor in-place or add wrapper layer
- Batch API batch sizes and polling intervals
- Order of execution within pipeline phases

### Claude's Discretion
- Exact execution sequence and parallelism within pipeline phases
- Batch sizes for Batch API submissions
- How to handle partial Batch API failures
- Synthetic prompt template adjustments for rejection examples
- Taxonomy minimum thresholds per category

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pipeline Scripts (to be executed with hardening)
- `scripts/phase1_clone.py` — Repo cloning, needs checkpoint integration
- `scripts/phase1_extract.py` — PHP function extraction, needs checkpoint integration
- `scripts/phase1_judge.py` — PHPCS pre-filter + Claude judge, needs utils.py integration (extract_json, call_with_backoff, checkpoints, Batch API)
- `scripts/phase2_gap_analysis.py` — Coverage gap analysis against taxonomy
- `scripts/phase2_mutate.py` — Automated contrastive pair mutation
- `scripts/phase2_generate.py` — Synthetic generation, needs utils.py integration + rejection example support
- `scripts/phase2_judge.py` — Synthetic judgment, needs utils.py integration
- `scripts/phase2_judge_dataset.py` — Judge training data, needs utils.py integration (rate limiting fix confirmed)
- `scripts/phase3_cot.py` — CoT reasoning, needs utils.py integration
- `scripts/export_dataset.py` — Final export, needs 40/60 ratio update

### Phase 1 Outputs (available for integration)
- `scripts/utils.py` — 9 functions: extract_json, call_with_backoff, load_checkpoint, save_checkpoint, batch_or_direct, make_batch_request, submit_batch, poll_batch, parse_batch_results
- `scripts/preflight.py` — Pre-flight validation (PHPCS, PHP CLI, API key)
- `config/repos.yaml` — 56 repos (1 core + 49 plugins + 6 themes) with quality_tier

### Configuration
- `config/judge_system.md` — Judge system prompt (needs security dimension < 5 = auto-FAIL update)
- `config/taxonomy.yaml` — Concept taxonomy with minimum coverage targets
- `config/synthetic_prompts.yaml` — Generation templates by gap tag (needs rejection example templates)

### Source Data
- `/home/robert_li/Desktop/data/wp-finetune-data/wp_top1000_plugins_final.csv` — Fallback source for additional repos if needed

### Known Issues
- `.planning/codebase/CONCERNS.md` — phase2_judge_dataset.py has no rate limiting (PIPE-03 fix needed)
- `.planning/research/PITFALLS.md` — Parse failure stubs, checkpoint gaps

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/utils.py` (334 lines): extract_json (4-strategy fallback), call_with_backoff (exponential + retry-after), checkpoint save/load (atomic rename), Batch API routing (threshold=50)
- `scripts/preflight.py` (85 lines): Pre-execution validation
- `scripts/csv_to_repos.py` (276 lines): Can be re-run with different filters if more repos needed
- `config/repos.yaml` (367 lines): 56 repos ready for cloning

### Established Patterns
- All pipeline scripts use `anthropic.Anthropic()` from env var
- Rate limiting via `time.sleep(REQUEST_INTERVAL)` — needs replacement with `call_with_backoff()`
- JSON parsing via brittle string splitting — needs replacement with `extract_json()`
- Output stored as JSON files per-repo in `{phase}_extraction/output/` or `{phase}_synthetic/output/`

### Integration Points
- `phase1_clone.py` reads `config/repos.yaml` via yaml.safe_load
- `phase1_judge.py` writes to `phase1_extraction/output/passed/` and `failed/`
- `phase2_gap_analysis.py` reads passed functions + `config/taxonomy.yaml`
- `phase2_generate.py` reads style anchors from Phase 1 passed output
- `export_dataset.py` reads all Phase 1-3 output directories

</code_context>

<specifics>
## Specific Ideas

- User wants the model to "push back harder" — proactively add security measures even when prompts don't mention them
- Rejection examples: "write a form handler" → model adds nonces + explains CSRF risk
- The dual-mode architecture means the generation pathway should inherently avoid patterns the judge pathway would flag
- 40/60 gen/judge emphasis gives the model stronger critic capability
- Aggressive security scoring (any security dim < 5 = auto-FAIL) raises the floor on training data quality

</specifics>

<deferred>
## Deferred Ideas

- Adversarial examples from Phase D4 fed back into training data as feedback loop — Phase D4 hasn't run yet, belongs in v2 training cycle
- The dual-mode architecture insight (generation avoids patterns judge would flag) is a training-time concern for Phase 3 config

</deferred>

---

*Phase: 02-dataset-production*
*Context gathered: 2026-03-26*
