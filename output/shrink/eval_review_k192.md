# Independent review: shrink-k192.Q6_K.gguf (from wp-judge-v4-pruned-k224.Q6_K.gguf)

gguf-py only (~/work/llama.cpp/gguf-py), memmap reads, no GPU.

1. Header - PASS. expert_count 224->192; expert_used_count 8==8; block_count 40; arch qwen35moe. Key sets identical; the only differing value is expert_count. Tokenizer, chat template, rope etc. unchanged.
2. Tensors - PASS. The same 733 names in the same order. All 573 non-expert tensors (1.995 GB, incl. 160 *shexp*) have matching sha256, type and shape.
3. Expert tensors - PASS. ffn_{gate,up,down}_exps and ffn_gate_inp go 224->192 with the same quant type. This arch has no exp_probs_b. Layers 0, 19 and 39: every kept slot is byte-equal to its source slot in all 4 tensors. Removed GGUF slots: L0 [9,10,11,14,20,21,34,35,39,50,59,66,77,81,83,86,98,103,104,118,119,124,128,175,178,187,196,200,215,216,217,223]; L19 [5,23,29,35,36,38,47,54,86,88,93,99,109,112,116,120,121,122,132,138,144,152,166,176,180,183,186,193,197,198,214,221]; L39 [12,13,15,23,41,43,45,58,69,92,99,104,108,111,112,114,136,141,145,157,159,179,184,189,191,197,205,210,215,216,217,218].
4. Selection - PASS. Direction comes from build_uniform_keep_mask, which the v4 scripts call: argsort(-score), so a higher AIMER score is kept. Slot order follows sorted(kept_idx) in prune_apply_physical_v4. Protected mask sha256 matches the manifest. Recomputed keep sets (protected plus top non-protected scores, no ties at the cutoff) match the actual kept sets in all 40 layers (router-row matching). All protected experts are retained (max 98/layer).
5. Size - PASS. 25,200,652,096 -> 21,887,151,936 B. The difference, 3,313,500,160 B, is exactly 32/224 of the expert bytes (23,194,501,120).
