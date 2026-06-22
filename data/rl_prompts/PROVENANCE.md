# RL Prompt Pool Provenance

## Purpose

Auditable lineage for the RL rollout prompt pools consumed by Phase 9 (GSPO Training).
Every prompt in these pools derives from the Phase-4.2 audited training corpus.

This document satisfies the T-09-POISON and T-09-LEAK threat mitigations
defined in `09-01-PLAN.md`.

## Source Files

| File | Role |
|------|------|
| `data/reasoning_dataset/openai_train.jsonl` | Phase-4.2 canonical train corpus (passed boundary/security review, vendorfilter applied) |
| `data/reasoning_dataset/openai_val.jsonl` | Held-out val set — used for leakage guard ONLY, no rows emitted |

## Split Rule

User-turn content routing:
- Starts with `<wp_gen>`   → `wp_gen_train.jsonl`
- Starts with `<wp_judge>` → `wp_judge_train.jsonl`
- Neither tag              → excluded (replay rows without tags)

## Row Counts

| Metric | Count |
|--------|-------|
| Train rows read | 563 |
| Tagged `<wp_gen>` (before dedup) | 68 |
| Tagged `<wp_judge>` (before dedup) | 482 |
| Untagged / replay rows skipped | 8 |
| Dedup dropped (sha256 collision) | 5 |
| Val-leakage dropped | 0 |
| **wp_gen pool output** | **68** |
| **wp_judge pool output** | **482** |

## Val-Set Leakage Guard (T-09-LEAK)

Val rows loaded for leakage check: 141

Every emitted prompt's user-content sha256 was checked against the full
`openai_val.jsonl` sha256 set before emission.

**Assertion: NO val-set user-content sha256 appears in either RL prompt pool.**
Val-leakage dropped rows: 0

## Deduplication

Dedup is performed on sha256(user_content) across BOTH pools (gen + judge share
a single seen-hash set). A duplicate appearing in the same or different pool is
dropped; only the first-seen instance is emitted.

Dedup dropped: 5

## Output File Checksums

| File | SHA-256 |
|------|---------|
| `data/rl_prompts/wp_gen_train.jsonl` | `75df101a66965dad6688406fa625084044850aa0e9275f765695ade4ea487ee0` |
| `data/rl_prompts/wp_judge_train.jsonl` | `4a5d4b301d1b11bfcf3bac0076fedce18eccbd496e57d38d3dc72dd62d3f48a1` |

## Training-Data-Poisoning Mitigation (T-09-POISON)

- Sources: ONLY the audited Phase-4.2 train corpus (vendor-filtered,
  boundary/security reviewed, per STATE.md Phase 4.2 entry).
- Excluded: `openai_val.jsonl` (held-out), `*.pre_vendorfilter.*` backups
  (vendor-contaminated pre-filter), any synthetic or unaudited prompt source.
- Lineage: fully traceable to the single source file listed above.
- Idempotent: re-running this script with the same source produces byte-identical output.

## Schema

Both output files use OpenAI chat format with an **empty assistant turn**:

```json
{"messages": [{"role": "user", "content": "<wp_gen> ..."}, {"role": "assistant", "content": ""}]}
```

Completions are generated at RL sampling time — the assistant turn is never pre-filled.
