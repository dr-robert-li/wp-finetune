#!/usr/bin/env bash
# 04.3-03 merge-vs-training discriminator — FULL unattended run under multi-user.target.
# The in-process 30B bf16 load peaks ~100-103 GiB on the GB10 unified pool; with GNOME up there
# is no floor that both completes the load AND protects the host (proven: 3 loaders all trip).
# Dropping to multi-user.target frees GNOME's ~6 GiB so the load can complete (the Jun-5 pre-quant
# completed there). This script runs Task 1 (binding-dryrun gate) -> Task 2 (3 bf16 arms, SEQUENTIAL,
# never co-resident) -> Task 3 (Wilson-CI verdict), then restores graphical.target.
#
# RUN DETACHED AS ROOT (survives the graphical drop):
#   sudo systemd-run --unit=disc043 --collect --property=IgnoreOnIsolate=yes \
#     bash /home/robert_li/Desktop/projects/wp-finetune/scripts/_discriminator_multiuser.sh
# Watch from a TEXT TTY (Ctrl+Alt+F3, survives the isolate):
#   journalctl -u disc043 -f ; tail -f logs/discriminator_multiuser.log
# Desktop drops, the job runs ~1.5-2 hr (3 sequential captures), then graphical.target returns.
# IgnoreOnIsolate=yes is REQUIRED or `systemctl isolate` kills this transient unit itself.
set -u
cd /home/robert_li/Desktop/projects/wp-finetune || exit 2
mkdir -p logs output/format_stability/discriminator
DISC=output/format_stability/discriminator
VAL=data/reasoning_dataset/openai_val.jsonl
MERGED=models/qwen3-30b-wp-30_70-reasoning-merged
BASE=models/qwen3-30b-wp-30_70-merged-v2
CKPT=adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72
LOG=logs/discriminator_multiuser.log
exec >>"$LOG" 2>&1
# No GNOME to OOM-cascade once isolated; 10 GiB backstops the bare-kernel floor while giving the
# ~103-108 GiB bf16 peak room to complete (121 - 108 = ~13 GiB trough > 10).
ARM_TRIP=10240

stamp(){ date -u +%FT%TZ; }
avail_mib(){ free -m | awk '/^Mem:/{print $7}'; }
restore_graphical(){
  echo "[$(stamp)] restoring graphical.target ..."
  systemctl isolate graphical.target \
    || echo "[$(stamp)] WARN: restore failed — run 'sudo systemctl isolate graphical.target' manually"
}
trap restore_graphical EXIT
tripped(){ grep -q 'tripped=1' "logs/$1_watchdog.log" 2>/dev/null; }
wait_for_ram(){ # $1 target MiB, $2 timeout sec — non-fatal (watchdog backstops the next load)
  local tgt="$1" to="${2:-180}" t=0
  sync; echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
  while [ "$(avail_mib)" -lt "$tgt" ]; do
    sleep 3; t=$((t+3)); sync; echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
    if [ "$t" -ge "$to" ]; then
      echo "[$(stamp)] WARN wait_for_ram timeout avail=$(avail_mib) < $tgt — proceeding (watchdog backstops)"
      return 1
    fi
  done
  echo "[$(stamp)] ram recovered avail=$(avail_mib)MiB (>= $tgt)"
}

echo "[$(stamp)] ===== DISCRIMINATOR START (multi-user.target) ====="
docker ps --format '{{.Names}}' | grep -q '^unsloth-headless$' \
  || { echo "[$(stamp)] ERROR: unsloth-headless container not running; abort before isolate"; exit 3; }
echo "[$(stamp)] pre-isolate avail=$(avail_mib)MiB"
echo "[$(stamp)] isolating multi-user.target (dropping GNOME) ..."
systemctl isolate multi-user.target
sleep 6; sync; echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
echo "[$(stamp)] post-isolate avail=$(avail_mib)MiB — starting Task 1"

# ---------------- Task 1: bf16 binding-dryrun gate (also the isolated-load feasibility test) -------
rm -f "$DISC/binding_dryrun.md"
TRIP_MIB=$ARM_TRIP bash scripts/_mem_watchdog.sh disc_binding \
  scripts.checkpoint_parse_check --binding-dryrun --no-4bit --base "$BASE" --checkpoint-dir "$CKPT"
if tripped disc_binding || [ ! -s "$DISC/binding_dryrun.md" ] || ! grep -q '^## VERDICT:' "$DISC/binding_dryrun.md"; then
  echo "[$(stamp)] LOAD INFEASIBLE EVEN ISOLATED — binding-dryrun did not complete "
  echo "[$(stamp)]   (watchdog tripped or no VERDICT written). This is an INFRA blocker, NOT a binding verdict."
  { echo "LOAD_INFEASIBLE_EVEN_ISOLATED"; echo "disc_binding watchdog:"; tail -3 logs/disc_binding_watchdog.log; } > "$DISC/INFRA_BLOCKER.txt"
  exit 10
fi
if grep -q '^## VERDICT: BINDING_FAILED' "$DISC/binding_dryrun.md"; then
  echo "[$(stamp)] binding guard => BINDING_FAILED; skipping captures, recording verdict"
  python3 scripts/_discriminator_verdict.py --binding-failed
  exit 0
fi
echo "[$(stamp)] binding guard => BOUND; proceeding to the three captures"

# ---------------- Task 2: three bf16 arms, SEQUENTIAL (never co-resident) --------------------------
# ARM 1 — merged vLLM bf16. --min-parseable-rate 0.0 so a terse/low-parseable collapse (the thing we
# MEASURE) does not abort the capture; histogram is a sibling of --out -> rename to the arm name.
wait_for_ram 110000 120
TRIP_MIB=$ARM_TRIP bash scripts/_mem_watchdog.sh disc_arm1_vllm \
  scripts.capture_reasoning_responses --model-dir "$MERGED" --dataset "$VAL" \
  --include-streams cot,ctf --limit 120 --max-tokens 2048 --gpu-mem-util 0.55 \
  --min-parseable-rate 0.0 --out "$DISC/merged72_vllm.jsonl"
mv -f "$DISC/capture_format_histogram.json" "$DISC/merged72_vllm_histogram.json" 2>/dev/null \
  || echo "[$(stamp)] WARN: ARM 1 histogram (capture_format_histogram.json) not found"

# ARM 2 — merged Unsloth bf16 (merge-math applied, no load_adapter). Let it FULLY exit before ARM 3.
wait_for_ram 110000 300
TRIP_MIB=$ARM_TRIP bash scripts/_mem_watchdog.sh disc_arm2_merged \
  scripts.checkpoint_parse_check --base "$MERGED" --no-adapter --no-4bit \
  --include-streams cot,ctf --max-new-tokens 2048 --n 120 --val-jsonl "$VAL" \
  --out "$DISC/merged72_unsloth_histogram.json"

# ARM 3 — unmerged Unsloth bf16 (load_adapter on merged-v2; re-asserts the runtime binding guard).
wait_for_ram 110000 300
TRIP_MIB=$ARM_TRIP bash scripts/_mem_watchdog.sh disc_arm3_unmerged \
  scripts.checkpoint_parse_check --base "$BASE" --checkpoint-dir "$CKPT" --no-4bit \
  --include-streams cot,ctf --max-new-tokens 2048 --n 120 --val-jsonl "$VAL" \
  --out "$DISC/unmerged72_unsloth_histogram.json"

# ---------------- Task 3: Wilson-CI verdict (CPU, host python) -------------------------------------
wait_for_ram 60000 120
H1="$DISC/merged72_vllm_histogram.json"; H2="$DISC/merged72_unsloth_histogram.json"; H3="$DISC/unmerged72_unsloth_histogram.json"
if [ -s "$H1" ] && [ -s "$H2" ] && [ -s "$H3" ]; then
  python3 scripts/_discriminator_verdict.py
elif [ ! -s "$H3" ] && grep -q 'BINDING_FAILED' logs/disc_arm3_unmerged.log 2>/dev/null; then
  echo "[$(stamp)] ARM 3 re-assertion => BINDING_FAILED; recording verdict"
  python3 scripts/_discriminator_verdict.py --binding-failed
else
  echo "[$(stamp)] INFRA: a capture arm did not produce its histogram (H1=$([ -s "$H1" ]&&echo ok||echo MISSING) H2=$([ -s "$H2" ]&&echo ok||echo MISSING) H3=$([ -s "$H3" ]&&echo ok||echo MISSING)) — no verdict written; investigate logs/disc_arm*.log"
  { echo "CAPTURE_INCOMPLETE"; echo "H1=$([ -s "$H1" ]&&echo ok||echo MISSING) H2=$([ -s "$H2" ]&&echo ok||echo MISSING) H3=$([ -s "$H3" ]&&echo ok||echo MISSING)"; } > "$DISC/INFRA_BLOCKER.txt"
  exit 11
fi
echo "[$(stamp)] ===== DISCRIMINATOR END ====="
exit 0
