# shrink_mem.sh review

1. SHOULD-FIX L21-24: readiness loop falls through after 300 tries and measures anyway. Add READY flag + exit 3 (as shrink_eval.sh L27).
2. SHOULD-FIX L25-35: empty PROMPT or failed curl (-f, 5xx, timeout) still writes memory.json. Add `[ -n "$PROMPT" ] || exit 4` and `wait $CPID || exit 4`.
3. SHOULD-FIX L20/36: non-numeric used_memory (e.g. `[N/A]`) only prints a test error and leaves PEAK=0 (tested). Exit if PEAK is still 0 after the final sample.
4. NIT L38-40: printf JSON breaks on quotes in EXTRA/GGUF and on empty RAM if the server died. Use python json.dump (shrink_eval.sh L38-52).
5. NIT: no `[ -f "$GGUF" ]` guard.

Checked OK: $! is llama-server itself (tested), awk parse, integer max, /health returns 503 until loaded, CUDA pool never shrinks so the final sample catches the peak, request matches _judge_create, VmHWM read before kill, empty EXTRA under set -u (bash 5.2), port 8098 free.
