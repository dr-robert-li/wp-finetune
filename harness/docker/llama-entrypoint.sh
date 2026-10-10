#!/bin/sh
# Download the GGUF on first run (resumable), then exec llama-server. Extra args pass through.
set -e
REPO=${WPJUDGE_MODEL_REPO:-iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf}
FILE=${WPJUDGE_MODEL_FILE:-wp-judge-v4-pruned-k224.IQ4_XS.gguf}
case "$FILE" in */*|..*|"") echo "bad WPJUDGE_MODEL_FILE: $FILE" >&2; exit 2;; esac
M=/models/$FILE
if [ ! -f "$M" ]; then
  echo "downloading $REPO/$FILE to $M" >&2
  curl -fL -C - --retry 5 --retry-delay 10 -o "$M.part" "https://huggingface.co/$REPO/resolve/main/$FILE"
  mv "$M.part" "$M"
fi
P=${LLAMA_PARALLEL:-1}  # >1 batches decoding: faster, not reproducible (T5)
# No -ctk/-ctv q8_0: the small file needs the default f16 KV cache.
exec /app/llama-server -m "$M" --host 0.0.0.0 --port 8080 -ngl 999 --jinja \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --parallel "$P" -c $((P * ${LLAMA_SLOT_CTX:-8192})) "$@"
