# Independent review: f16 rebuild receipt (2026-10-09)

1. PASS. The sha256 of rebuild-check.Q6_K matches the shipped Q6_K: df815714...c729. It also matches the HF LFS oid for wp-judge-v4-pruned-k224.Q6_K.gguf, and all three are 25200652096 bytes.
2. PASS. The f16 is 61313087616 bytes, the same size as in conversion_receipt_v4. It is qwen35moe with block_count 40 and expert_count 224. It has 733 tensors and no mtp/nextn tensors or keys. 432 are F16 and 301 are F32, and no tensor over 1M elements is anything but F16. The F32 tensors are the norms, ssm_*, ffn_gate_inp and ffn_gate_inp_shexp.
3. PASS. The sha256 is 33e428f4...d36c, as the receipt says.
4. PASS. The manifest has promoted = wp-judge-v4-s1-ep3. 26-02 builds the pruned checkpoint from judge-v4-s1-merged, and the ship decision compares s1 against s1. 27-02 converts that pruned-k224 checkpoint. 27-02 does not name s1 itself, so the s1 link comes from 26-02.
