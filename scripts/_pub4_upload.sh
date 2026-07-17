#!/usr/bin/env bash
# PUB4-01 upload runner v4: SEQUENTIAL per-file `hf upload`.
# upload-large-folder deadlocked twice on this host (10 workers wedged at
# pre-upload, zero io) even with Xet disabled + hub 1.23 — single-file
# `hf upload` probe worked end to end, so ship sequentially.
# Per-file: 3 attempts + stall watchdog (kill if no io growth for 5 min).
# Allowlist-only: iterates output/pkg-v4/pub4_upload_manifest.json.
# Target repo: iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf (new v4 repo).
# Log: logs/hf_upload_27.log
set -uo pipefail
cd /home/robert_li/Desktop/projects/wp-finetune
unset HF_XET_HIGH_PERFORMANCE
export HF_HUB_DISABLE_XET=1

# sum wchar over a process and its descendants (the hf wrapper forks a python worker)
tree_wchar() {
  local total=0 p w
  for p in $1 $(pgrep -P "$1" 2>/dev/null); do
    w=$(awk '/^wchar/{print $2}' "/proc/$p/io" 2>/dev/null) || w=0
    total=$((total + ${w:-0}))
  done
  echo "$total"
}

upload_one() { # repo local remote
  local repo=$1 local_f=$2 remote=$3 i wpid last cur stall rc
  for i in 1 2 3; do
    echo "[$(date -u +%FT%TZ)] upload $remote -> $repo attempt $i ($(stat -c%s "$local_f") bytes)"
    hf upload "$repo" "$local_f" "$remote" &
    wpid=$!
    last=0; stall=0
    while kill -0 "$wpid" 2>/dev/null; do
      sleep 30
      cur=$(tree_wchar "$wpid")
      if [ "$cur" -le "$last" ]; then stall=$((stall + 1)); else stall=0; fi
      last=$cur
      if [ "$stall" -ge 10 ]; then
        echo "[$(date -u +%FT%TZ)] STALL: $remote no io growth 5min, killing attempt $i"
        pkill -TERM -P "$wpid" 2>/dev/null; kill -TERM "$wpid" 2>/dev/null
        sleep 5; pkill -KILL -P "$wpid" 2>/dev/null; kill -KILL "$wpid" 2>/dev/null
        break
      fi
    done
    wait "$wpid"; rc=$?
    [ "$rc" -eq 0 ] && { echo "[$(date -u +%FT%TZ)] DONE $remote"; return 0; }
    echo "[$(date -u +%FT%TZ)] attempt $i failed (rc=$rc), retry in 30s"
    sleep 30
  done
  echo "[$(date -u +%FT%TZ)] FATAL: $remote failed after 3 attempts"
  return 1
}

echo "[$(date -u +%FT%TZ)] PUB4-01 sequential upload start"

# iterate manifest allowlist: "repo_id<TAB>local<TAB>remote" per line
FAIL=0
while IFS=$'\t' read -r repo local_f remote; do
  upload_one "$repo" "$local_f" "$remote" || { FAIL=1; break; }
done < <(python3 -c "
import json
man = json.load(open('output/pkg-v4/pub4_upload_manifest.json'))
for r in man['repos']:
    for f in r['files']:
        print(f\"{r['repo_id']}\t{f['path']}\t{f['repo_path']}\")
")

[ "$FAIL" = 0 ] || { echo "[$(date -u +%FT%TZ)] PUB4-01 upload FAILED"; exit 1; }
echo "[$(date -u +%FT%TZ)] PUB4-01 upload ALL DONE"
