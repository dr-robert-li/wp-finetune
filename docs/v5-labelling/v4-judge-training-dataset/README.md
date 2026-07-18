# v4-judge-training-dataset/ — overlap-check + label-format reference

Copies of the v4 judge's datasets (sources of truth: `data/reasoning_dataset/` at repo root).

| File | Rows | Role |
|---|---|---|
| `openai_train_relabel_v1.jsonl` | 563 (482 wp_judge) | v4 SFT training set — new EVAL items must not overlap it |
| `openai_val.jsonl` | 141 (121 wp_judge) | held-out eval behind every published rho — new TRAINING items must NEVER overlap it (hard gate) |

Dedup/contamination procedure + disposition rules: EXTRACTION_GUIDE.md §4.
Label-format exemplars (per-dimension prose + verdict): §5.
