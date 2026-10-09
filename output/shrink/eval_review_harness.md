# Shrink harness review (independent)

1. BLOCKER scripts/sieve_capture_judge_http.py:41,84 - client concurrency 4 (not overridable) -> random arrival order at the single --parallel 1 slot; llama-server cache_prompt=true (and --cache-ram) reuses the KV prefix, which llama.cpp's server README says is not bit-exact. baseline_repeat may differ. Fix: add --no-cache-prompt to shrink_eval.sh:15-16 and a --concurrency flag, set to 1.
2. BLOCKER scripts/shrink_eval.sh:41 - `eval time =.*?` also matches "prompt eval time" lines (synthetic test gave [1885.94, 50.0]), so the speed median is wrong. Fix: `(?m)^\s+eval time =`.
3. BLOCKER scripts/shrink_run_all.sh:98-99 - prune reads "$SRC"; with SRC_GGUF=f16 it writes an f16-sized file named .Q6_K. Fix: prune from $SHIP (or quantize after pruning).
4. SHOULD-FIX shrink_eval.sh:44 - with --no-mmap on CUDA, CPU-held weights go into the "CUDA_Host" buffer, so cpu_weights_mib reads 0 for rt_cpumoe. Fix: `(CPU\w*|CUDA_Host) model buffer size`.
5. SHOULD-FIX shrink_eval.sh:18-31 - continues when the server never got ready or the capture failed, still writes resources.json, and run_all then skips that variant for good. Fix: exit non-zero unless 121/121 ok.
6. SHOULD-FIX scripts/shrink_score.py:42-45 - a response truncated before <judge_output> gets a prose-derived score (82.0 in test) instead of parse_fail. Fix: require _format=="json" or report format counts.
7. SHOULD-FIX shrink_score.py:44 - JSON with only overall_score -> counted as parse_fail. Fix: accept when "overall" in p.
8. SHOULD-FIX shrink_eval.sh - 4 queued requests can exceed the openai client's 600s timeout on rt_cpumoe; retries add requests to the queue. Fix: concurrency 1 (item 1).
9. NIT shrink_run_all.sh:86-97 - TAG also renames baseline/rt variants, so they rerun with no change. Fix: leave $T off those.
10. NIT shrink_eval.sh:40 - `tg` unused; the "hi" warmup requests are counted in ev.
11. NIT shrink_score.py:65 - a NaN in the bootstrap breaks the sort. Fix: np.nanpercentile.

Verified OK: tensor-type regex vs qwen35moe names (regex_search); -fa on / -ctk / --n-cpu-moe flags valid; index alignment; verdict regex; score direction (k224 mask reproduces from scores); protected mask sha matches manifest; calib from train split only.

self-checks: shrink_score.py --self-check OK (rc0); shrink_prune_gguf.py --self-check OK (rc0); bash -n both shell scripts OK; shell scripts have no self-check.
