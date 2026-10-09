# Shrink harness re-review (v2)

Fixed: 1/8 (--concurrency 1, --no-cache-prompt, --cache-ram 0 all valid), 3, 4, 5, 6/7 (synthetic: truncated -> not_json=1, overall-only parsed, empty -> parse_fail=1), 9 (run() tag logic correct for SHIP vs tagged files), 11.

NOT fixed / new defects:
1. BLOCKER scripts/shrink_eval.sh:45 - the new `(?m)^\s+eval time` never matches. SLT_INF prefixes every timing line with "slot print_timing: id  0 | task N |" (tools/server/server-common.h:20), so ev=[] and gen_tok_s_median is always None (synthetic test: []). Fix: `\|\s+eval time =\s*[\d.]+ ms /\s*(\d+) tokens` (tested: [('500','50.00')]; "| prompt eval" is still excluded).
2. NIT scripts/shrink_run_all.sh:105-107 - the f16 pruned intermediate (~60 GB) is never deleted. Fix: rm "$tmp" after a successful quantize.
3. NIT shrink_run_all.sh:107 - the f16-pruned Q6_K is quantized without an imatrix. Confirm the shipped Q6_K was also built without one, or the comparison is not like-for-like.

self-checks: shrink_score.py OK; shrink_prune_gguf.py OK; bash -n both OK; capture --help shows --concurrency.
