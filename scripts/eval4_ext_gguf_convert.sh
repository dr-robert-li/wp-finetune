#!/usr/bin/env bash
# Phase 23-02 extension: convert a merged HF judge-v4 checkpoint directly to
# Q8_0 GGUF (mirrors the v3 ship path -- JOURNAL.md: "convert_hf_to_gguf.py
# --outtype q8_0 produced a real Q8_0 of the single-seed v1.3 judge"), then
# sanity-check the GGUF block count against the source config's
# num_hidden_layers. Text-pipeline conversion drops the VL vision tower --
# expected (no --mmproj flag passed).
#
# Usage: eval4_ext_gguf_convert.sh <merged_hf_dir> <out_gguf_path> [outtype=q8_0] [extra_convert_arg]
#
# [extra_convert_arg] (27-02 addition): pass "--no-mtp" for checkpoints whose MTP/nextn
# layer was NOT pruned in lockstep with the trunk (e.g. Phase 26 surgery deliberately left
# mtp.*/shared_expert.* untouched at their original expert count). GGUF has one GLOBAL
# expert_count field -- llama.cpp's loader hard-asserts every ffn_gate_inp.weight matches it,
# so a trunk pruned to 224 experts + an unpruned 256-expert MTP block in the SAME GGUF fails to
# load ("check_tensor_dims: tensor 'blk.N.ffn_gate_inp.weight' has wrong shape"). --no-mtp
# drops the mtp.* tensors and sets block_count=num_hidden_layers exactly (no mixed-count
# block) -- confirmed in ~/llama.cpp/conversion/qwen.py's _Qwen35MtpMixin. Safe: the eval
# harnesses in this repo never exercise MTP/speculative decoding.
set -euo pipefail
ROOT=/home/robert_li/Desktop/projects/wp-finetune
LLAMACPP=~/llama.cpp
MERGED="$1"; OUT="$2"; OUTTYPE="${3:-q8_0}"; EXTRA="${4:-}"
cd "$ROOT"

[ -d "$MERGED" ] || { echo "MISSING merged dir $MERGED"; exit 2; }
mkdir -p "$(dirname "$OUT")"

echo "[convert] llama.cpp build:"
"$LLAMACPP/build/bin/llama-cli" --version 2>&1 | head -2
if [ -f "$OUT" ]; then
  echo "[convert] $OUT already exists -- skipping conversion, re-running sanity check only"
else
  echo "[convert] $MERGED -> $OUT (--outtype $OUTTYPE${EXTRA:+ $EXTRA})"
  python3 "$LLAMACPP/convert_hf_to_gguf.py" "$MERGED" --outtype "$OUTTYPE" --outfile "$OUT" ${EXTRA}
fi

echo "[convert] block-count + expert-count sanity check vs safetensors index / config.json"
python3 -c "
import json, sys
from gguf import GGUFReader
merged, out, no_mtp = '$MERGED', '$OUT', '$EXTRA' == '--no-mtp'
cfg = json.load(open(f'{merged}/config.json'))
tc = cfg.get('text_config', cfg)
# block_count includes the MTP layer(s) UNLESS --no-mtp excluded them (see header comment)
expected = tc['num_hidden_layers'] + (0 if no_mtp else tc.get('mtp_num_hidden_layers', 0))
# expert count: hard subscript, not .get -- a missing key must raise, never silently skip (T-27-01)
expected_experts = tc['num_experts']
r = GGUFReader(out)
bc = None
ec = None
for f in r.fields:
    if f.endswith('.block_count'):
        fld = r.fields[f]
        bc = int(fld.parts[fld.data[0]][0])
    if f.endswith('.expert_count'):
        fld = r.fields[f]
        ec = int(fld.parts[fld.data[0]][0])
print(f'expected block_count (num_hidden_layers + mtp)={expected} gguf_block_count={bc}')
assert bc == expected, f'BLOCK COUNT MISMATCH: gguf={bc} vs safetensors-index/config={expected}'
print('[convert] block-count sanity: PASS')
print(f'expected expert_count (config text_config.num_experts)={expected_experts} gguf_expert_count={ec}')
assert ec == expected_experts, f'EXPERT COUNT MISMATCH: gguf={ec} vs config={expected_experts}'
print(f'[convert] expert-count sanity: PASS ({ec})')
"
ls -la "$OUT"
echo "[convert] DONE $OUT"
