#!/usr/bin/env bash
# Probe ANY tinker sampler checkpoint: export -> merge -> wp-bench vs cached v1.2 0.4616.
# Generalization of _rlev01_probe_ckpt.sh (which hardcoded the seedA RUN id).
# Idempotent: skips export/merge if already present.
# Usage: _rlev01_probe_any.sh <tinker-sampler-path> <tag>
set -euo pipefail
cd /home/robert_li/Desktop/projects/wp-finetune
RUN="${1:?need tinker:// sampler path}"
TAG="${2:?need tag (e.g. v1.3)}"
EXP="models/tinker_export/${TAG}"
TAR="${EXP}/checkpoint.tar.gz/checkpoint.tar"
MERGED="models/_staging/qwen3-30b-wp-${TAG}-merged"
set -a; . ./.env; set +a; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
export PYTHONPATH=.
PY=.venv-tinker/bin/python

echo "=== [${TAG}] EXPORT ==="
mkdir -p "$EXP"
if [ -f "$TAR" ]; then echo "export cached: $TAR"; else
  $PY scripts/tinker_export_checkpoint.py --tinker-path "$RUN" --out "${EXP}/checkpoint.tar.gz"
fi

echo "=== [${TAG}] MERGE ==="
if [ -f "${MERGED}/config.json" ]; then echo "merge cached: $MERGED"; else
  $PY scripts/merge_tinker_v3.py --adapter-tar "$TAR" --base models/Qwen3-30B-A3B \
    --output-dir "$MERGED" --report "output/merge_${TAG}/merge_report.json"
fi

echo "=== [${TAG}] WP-BENCH ==="
$PY scripts/_rlev01_wpbench_ckpt.py --model-dir "$MERGED" --tag "${TAG}"
echo "=== [${TAG}] DONE ==="
