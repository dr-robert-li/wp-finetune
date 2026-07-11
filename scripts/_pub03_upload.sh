#!/usr/bin/env bash
# PUB-03 upload runner: gen staging dir via upload-large-folder (resumable),
# then the three judge Q8 GGUFs + card via hf upload. Allowlist-only — the
# staging dir contains exactly the manifest files (hardlinks).
# Log: logs/hf_upload_18_02.log
set -uo pipefail
cd /home/robert_li/Desktop/projects/wp-finetune
# Xet backend wedged on aarch64 (10 workers stuck "pre-uploading" 0 bytes for 1h, 2026-07-11).
# Force classic LFS multipart via hf_transfer instead.
unset HF_XET_HIGH_PERFORMANCE
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

GEN_REPO=iamchum/wp-qwen3-30b-a3b-wp-gen-v1.2
JUDGE_REPO=iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf
STAGE=models/_hf_stage_gen

echo "[$(date -u +%FT%TZ)] PUB-03 upload start (gen 57G + judge 3x30.2G)"

# --- gen repo (resumable large-folder upload, up to 3 attempts) ---
for i in 1 2 3; do
  echo "[$(date -u +%FT%TZ)] gen upload-large-folder attempt $i"
  hf upload-large-folder "$GEN_REPO" "$STAGE" --repo-type model && { GEN_OK=1; break; }
  GEN_OK=0
  echo "[$(date -u +%FT%TZ)] gen attempt $i failed (exit $?), retrying in 30s"
  sleep 30
done
[ "${GEN_OK:-0}" = 1 ] || { echo "[$(date -u +%FT%TZ)] FATAL: gen upload failed after 3 attempts"; exit 1; }
echo "[$(date -u +%FT%TZ)] gen upload COMPLETE"

# --- judge repo (3 GGUFs, each <50G LFS cap, resumable via retry) ---
for s in 0 1 2; do
  F=models/_gguf/wp-v1.3-judge-s$s.Q8_0.gguf
  for i in 1 2 3; do
    echo "[$(date -u +%FT%TZ)] judge s$s upload attempt $i"
    hf upload "$JUDGE_REPO" "$F" "wp-v1.3-judge-s$s.Q8_0.gguf" && { OK=1; break; }
    OK=0
    echo "[$(date -u +%FT%TZ)] judge s$s attempt $i failed, retrying in 30s"
    sleep 30
  done
  [ "${OK:-0}" = 1 ] || { echo "[$(date -u +%FT%TZ)] FATAL: judge s$s upload failed after 3 attempts"; exit 1; }
  echo "[$(date -u +%FT%TZ)] judge s$s COMPLETE"
done

hf upload "$JUDGE_REPO" output/packaging/hf_cards/judge_gguf_README.md README.md \
  && echo "[$(date -u +%FT%TZ)] judge README COMPLETE" \
  || { echo "[$(date -u +%FT%TZ)] FATAL: judge README upload failed"; exit 1; }

echo "[$(date -u +%FT%TZ)] PUB-03 upload ALL DONE"
