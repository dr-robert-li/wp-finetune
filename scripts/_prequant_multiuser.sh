#!/usr/bin/env bash
# ONE-TIME pre-quantization of merged-v2 (57GB bf16) -> bnb-4bit checkpoint, run under
# multi-user.target so there is NO GNOME OOM-victim and ~6 GiB more headroom. The ~100 GiB
# transient load peak is intrinsic to on-the-fly 4-bit quant of a bf16 base on unified memory
# (proven loader-independent: Unsloth, +expandable_segments, and transformers all peak ~100.6 GiB).
#
# RUN DETACHED AS ROOT (survives the graphical-target drop), e.g.:
#   sudo systemd-run --unit=prequant-mergedv2 --collect \
#     --property=IgnoreOnIsolate=yes \
#     bash /home/robert_li/Desktop/projects/wp-finetune/scripts/_prequant_multiuser.sh
# Watch from another TTY:  journalctl -u prequant-mergedv2 -f   (and tail logs/prequant_mergedv2*.log)
# The desktop will drop, the job runs (~6-8 min), then graphical.target is restored automatically.
#
# CRITICAL: --property=IgnoreOnIsolate=yes is REQUIRED. Without it, the `systemctl isolate
# multi-user.target` below stops EVERY unit not wanted by the new target — including this
# transient unit itself. The unit gets SIGTERM, the EXIT trap restores graphical.target, and
# the job dies in the same second WITHOUT ever pre-quantizing (observed 2026-06-04 13:38:51:
# "isolating multi-user.target ... restoring graphical.target ... Terminated"). IgnoreOnIsolate
# keeps this unit (and its cgroup: docker exec + watchdog children) alive across the isolate.
# Verify it survived: `systemctl is-active prequant-mergedv2` should stay 'active' through the
# isolate, and the log should reach a "post-isolate avail=..." line (proof it got past isolate).
# docker.service is WantedBy multi-user.target so the container + GPU survive.
#
# FALLBACK (if IgnoreOnIsolate still self-kills): from a root TTY, isolate manually FIRST, then
# run the quant body, then restore — no transient unit to kill:
#   sudo systemctl isolate multi-user.target
#   sudo TRIP_MIB=12288 bash scripts/_mem_watchdog.sh prequant_mergedv2_load \
#     scripts._tf_4bit_peak_probe --save models/qwen3-30b-wp-30_70-merged-v2-4bit
#   sudo systemctl isolate graphical.target
set -u
cd /home/robert_li/Desktop/projects/wp-finetune || exit 2
mkdir -p logs models
OUT=models/qwen3-30b-wp-30_70-merged-v2-4bit
LOG=logs/prequant_mergedv2.log
exec >>"$LOG" 2>&1

stamp() { date -u +%FT%TZ; }
restore_graphical() {
  echo "[$(stamp)] restoring graphical.target ..."
  systemctl isolate graphical.target || echo "[$(stamp)] WARN: failed to restore graphical.target — run 'sudo systemctl isolate graphical.target' manually"
}
trap restore_graphical EXIT

echo "[$(stamp)] ===== PREQUANT START ====="
echo "[$(stamp)] pre-isolate avail=$(free -m|awk '/^Mem:/{print $7}')MiB  out=$OUT"

# docker.service is WantedBy multi-user.target, so the unsloth-headless container + GPU survive.
if ! docker ps --format '{{.Names}}' | grep -q '^unsloth-headless$'; then
  echo "[$(stamp)] ERROR: unsloth-headless container not running; aborting before isolate"; exit 3
fi

echo "[$(stamp)] isolating multi-user.target (dropping GNOME) ..."
systemctl isolate multi-user.target
sleep 6
sync; echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
echo "[$(stamp)] post-isolate avail=$(free -m|awk '/^Mem:/{print $7}')MiB — starting pre-quant"

# Trip low (12 GiB): with the desktop gone the natural ~20.3 GiB bottom gains ~6 GiB headroom
# (-> ~26 GiB), so this should NOT fire; it is only a kernel-floor guard. No GNOME to cascade-kill.
TRIP_MIB=12288 bash scripts/_mem_watchdog.sh prequant_mergedv2_load \
  scripts._tf_4bit_peak_probe --save "$OUT"
RC=$?

echo "[$(stamp)] pre-quant watchdog rc=$RC"
if [ "$RC" -eq 0 ] && [ -d "$OUT" ]; then
  echo "[$(stamp)] SUCCESS — 4-bit checkpoint at $OUT:"
  ls -la "$OUT" | sed 's/^/[saved] /'
  du -sh "$OUT" | sed 's/^/[saved] /'
else
  echo "[$(stamp)] FAILED rc=$RC (tripped=$(grep -o 'tripped=[01]' logs/prequant_mergedv2_load_watchdog.log | tail -1)); checkpoint NOT created"
fi
echo "[$(stamp)] ===== PREQUANT END rc=$RC ====="
# trap restores graphical.target on exit
exit "$RC"
