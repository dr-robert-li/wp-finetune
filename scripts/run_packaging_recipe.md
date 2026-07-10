# Packaging recipe — turnkey quantization + serving (Phase 15, PKG-03/05)

Runs on the DGX vLLM/llama.cpp container path, not local transformers (local 30B load hits the
unified-memory wall — see `output/format_stability/discriminator/MEMORY-INVESTIGATION-bf16.md`).

## Inputs

- Gen model: `models/qwen3-30b-wp-30_70-reasoning-merged-v4`
- Judge (single-seed ship target): `models/_staging/qwen3-30b-wp-v1.3-merged`
- Gate 1 baseline: `output/packaging/gate1_bf16_baseline.json`
- Stop rule + bands: `output/packaging/pkg03_quantization_ladder.json`

## Q8 / Q6 / Q5 — GGUF path (Ollama, K-quants keep more bits on router/attn)

```bash
# 1. Provision toolchain (once)
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j

# 2. Convert bf16 HF -> GGUF f16 (streams from disk, no full GPU load)
python convert_hf_to_gguf.py MODEL_DIR --outtype f16 --outfile MODEL.f16.gguf

# 3. Quantize each tier
./build/bin/llama-quantize MODEL.f16.gguf MODEL.Q8_0.gguf Q8_0
./build/bin/llama-quantize MODEL.f16.gguf MODEL.Q6_K.gguf  Q6_K
./build/bin/llama-quantize MODEL.f16.gguf MODEL.Q5_K_M.gguf Q5_K_M
```

## Q4 — AWQ path only (uniform nf4 is a measured FAIL; do not use bnb)

```bash
pip install autoawq
# AWQ W4A16 with WordPress calibration data; protects salient router/attn weights
python -m awq.quantize --model MODEL_DIR --w_bit 4 --calib data/relabel_v1/... --out MODEL-awq
```

## Eval each tier against Gate 1 (±2pp gate)

```bash
# Serve via vLLM/Ollama container, then reuse the measured harness:
#   gen:   eval/run_wp_bench.py  -> compare wp_bench vs 0.4484 (floor 0.4284)
#   judge: scripts/relabel/eval_relabel.py -> compare rho vs 0.7554 ensemble / 0.7497 s1
# Stop at the lowest tier where BOTH stay within 2pp of Gate 1. Record in pkg03_quantization_ladder.json.
```

## PKG-05 — E2E validation on the shipped format

```bash
# Serve final GGUF via Ollama (or AWQ via vLLM), run 20 <wp_gen> + 20 <wp_judge> prompts,
# check coherent output + correct task-token routing. Reuse prompts: data/phase4_4/smoke_prompts.json
```

Uniform nf4 4-bit is excluded (`models/qwen3-30b-wp-30_70-merged-v2-4bit` = the collapse tombstone).
