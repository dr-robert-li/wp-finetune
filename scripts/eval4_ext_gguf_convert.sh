#!/usr/bin/env bash
# Phase 23-02 extension: convert a merged HF judge-v4 checkpoint directly to
# Q8_0 GGUF (mirrors the v3 ship path -- JOURNAL.md: "convert_hf_to_gguf.py
# --outtype q8_0 produced a real Q8_0 of the single-seed v1.3 judge"), then
# sanity-check the GGUF block count against the source config's
# num_hidden_layers. Text-pipeline conversion drops the VL vision tower --
# expected (no --mmproj flag passed).
#
# Usage: eval4_ext_gguf_convert.sh <merged_hf_dir> <out_gguf_path>
set -euo pipefail
ROOT=/home/robert_li/Desktop/projects/wp-finetune
LLAMACPP=~/llama.cpp
MERGED="$1"; OUT="$2"
cd "$ROOT"

[ -d "$MERGED" ] || { echo "MISSING merged dir $MERGED"; exit 2; }
mkdir -p "$(dirname "$OUT")"

echo "[convert] llama.cpp build:"
"$LLAMACPP/build/bin/llama-cli" --version 2>&1 | head -2
if [ -f "$OUT" ]; then
  echo "[convert] $OUT already exists -- skipping conversion, re-running sanity check only"
else
  echo "[convert] $MERGED -> $OUT (--outtype q8_0)"
  python3 "$LLAMACPP/convert_hf_to_gguf.py" "$MERGED" --outtype q8_0 --outfile "$OUT"
fi

echo "[convert] block-count sanity check vs safetensors index"
python3 -c "
import json, sys
from gguf import GGUFReader
merged, out = '$MERGED', '$OUT'
cfg = json.load(open(f'{merged}/config.json'))
tc = cfg.get('text_config', cfg)
# block_count includes the MTP layer(s): b9180+ converters export them (bartowski ships likewise)
expected = tc['num_hidden_layers'] + tc.get('mtp_num_hidden_layers', 0)
r = GGUFReader(out)
bc = None
for f in r.fields:
    if f.endswith('.block_count'):
        fld = r.fields[f]
        bc = int(fld.parts[fld.data[0]][0])
        break
print(f'expected block_count (num_hidden_layers + mtp)={expected} gguf_block_count={bc}')
assert bc == expected, f'BLOCK COUNT MISMATCH: gguf={bc} vs safetensors-index/config={expected}'
print('[convert] block-count sanity: PASS')
"
ls -la "$OUT"
echo "[convert] DONE $OUT"
