#!/usr/bin/env bash
# Probe one seedA checkpoint: export -> merge -> wp-bench vs cached v1.2 0.4616.
# Idempotent: skips export/merge if already present. Usage: _rlev01_probe_ckpt.sh <step>
set -euo pipefail
cd /home/robert_li/Desktop/projects/wp-finetune
STEP="${1:?need step number}"
RUN="tinker://9cb14129-f302-5c84-adf2-cc9ab92128a4:train:0/sampler_weights/step-${STEP}"
EXP="models/tinker_export/seedA-step${STEP}"
TAR="${EXP}/checkpoint.tar.gz/checkpoint.tar"
MERGED="models/_staging/qwen3-30b-wp-seedA-step${STEP}-merged"
set -a; . ./.env; set +a; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
export PYTHONPATH=.
PY=.venv-tinker/bin/python

echo "=== [step-${STEP}] EXPORT ==="
mkdir -p "$EXP"
if [ -f "$TAR" ]; then echo "export cached: $TAR"; else
  $PY scripts/tinker_export_checkpoint.py --tinker-path "$RUN" --out "${EXP}/checkpoint.tar.gz"
fi

echo "=== [step-${STEP}] MERGE ==="
if [ -f "${MERGED}/config.json" ]; then echo "merge cached: $MERGED"; else
  $PY scripts/merge_tinker_v3.py --adapter-tar "$TAR" --base models/Qwen3-30B-A3B \
    --output-dir "$MERGED" --report "output/merge_seedA_step${STEP}/merge_report.json"
fi

echo "=== [step-${STEP}] WP-BENCH ==="
$PY scripts/_rlev01_wpbench_ckpt.py --model-dir "$MERGED" --tag "seedA_step${STEP}"
echo "=== [step-${STEP}] DONE ==="
