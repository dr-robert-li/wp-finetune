# Phase-2 script review (static)

SHOULD-FIX
1. Lines 7-8: `pkill -f "llama-imatrix|llama-quantize"` and `"shrink_eval.sh iq4xs_im"` match any Claude `/bin/bash -c ... eval '...'` wrapper containing that text (confirmed: pgrep hit this reviewer's own shell). Line 6: if q4km capture fails (exit 4, no resources.json) phase 2 waits forever while phase 1 moves on. Fix: replace lines 6-8 with PID-based code:
   `kill -STOP 150847; while kill -0 1591293 2>/dev/null; do sleep 30; done; kill -KILL 150847; [ -f output/shrink/q4km/score.json ] || echo "q4km score missing"`
   (1591293 = shrink_eval q4km. A stopped driver cannot start iq4xs_im, so no imatrix or quantize to kill.)

PASS
- Race: TERM kills only the driver. The orphaned shrink_eval still writes score.json.
- Names: q5km_f16, shrink-*_f16.gguf, shrink_f16.imatrix. No collisions.
- prune_k192 file reused. prune_k160 built from SHIP. rt_* untagged.
- :8097: q4km server is killed before resources.json, and every eval has an EXIT trap.

NIT: the driver comment says `_f16src`, but the script uses `_f16`.

## Re-review: lines 6-13 (PID-based)
PASS. The zombie-aware loop exits once the eval is a zombie (Z) or gone. SIGKILL works on a stopped driver. The orphaned zombie is then reparented to systemd --user (PID 6795, the driver's parent and a subreaper) and reaped. Eval line 35 kills and waits on llama-server before resources.json, and its EXIT trap also covers it. Nothing is left on :8097.
NIT: if the driver has already exited, kill -STOP fails silently. That is harmless with no set -e, but launch phase 2 now.
