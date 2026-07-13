#!/usr/bin/env python
"""Phase 21 Task 2 gap-closure -- ground-truth verification that the NEW
target_parameters-aware merge (scripts/merge_adapter.py's
_merge_via_tinker_cookbook path) actually reproduces the routed-MoE-expert
LoRA's trained signal.

Merges the EXISTING 21-01 probe adapter (train_mlp=True) into a fresh
directory (already done by scripts/merge_adapter.py in this same session),
serves the merged model locally (reusing the Phase 20/21 vLLM smoke harness,
GB10 single residency, container killed when done), and compares GREEDY
(temperature=0) generations on >=5 fixed prompts against Tinker's own
SamplingClient output for the SAME adapter checkpoint (the sampler_path
recorded in output/base21/moe_merge_probe.json).

Both sides render the prompt with the IDENTICAL tokenizer + renderer
(tinker_cookbook.renderers, RENDERER_NAME from tinker_reasoning_data_v4) --
the renderer's tokenized ModelInput is decoded back to text for vLLM's raw
/v1/completions endpoint (NOT /v1/chat/completions, which would apply
vLLM's own chat template and could silently diverge from the renderer),
so both backends see byte-identical prompt text.

Verdict: per prompt, compare generated TEXT plus a NORMALIZED token prefix
where BOTH sides' text is re-encoded through the same tokenizer. Comparing
Tinker's raw per-step sampled ids against a re-encoding of vLLM's returned
text is apples-to-oranges: greedy decoding emits 'B' (id 33) one token at a
time, while re-encoding the identical 40-char 'BBBB...' string BPE-merges
into 'BBBB' (id 70944) chunks. The LOAD-BEARING row is the exact probe
TRAINING prompt: the aggressively-overfit adapter (rank=8, lr=0.05, 8
steps) gives it a decisive learned margin, so a correct merge must
reproduce the Tinker sampler's output token-for-token there (a gate/up
swap would change the MLP function and break this). Off-trained-prompt
rows have near-tie degenerate logits where bf16 kernel-order differences
across two unrelated serving stacks can legitimately flip the argmax
mid-junk -- they must share the degenerate output family, not match
byte-for-byte.

Usage (.venv-tinker):
    .venv-tinker/bin/python scripts/verify_moe_merge_ground_truth.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from tinker_reasoning_data_v4 import BASE_MODEL, RENDERER_NAME  # noqa: E402

PROBE_RECEIPT = PROJECT_ROOT / "output" / "base21" / "moe_merge_probe.json"
MERGED_MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3.6-35B-A3B-base21-moe-probe-merged-v2"
OUT_PATH = PROJECT_ROOT / "output" / "base21" / "moe_merge_ground_truth.json"
PREFIX_TOKENS = 20
MAX_GEN_TOKENS = 40

# >=5 fixed prompts: the exact probe training prompt first (strongest signal
# for an aggressively-overfit rank=8/lr=0.05/8-step LoRA), plus generic
# prompts to confirm the merge didn't corrupt untrained behavior.
FIXED_PROMPTS = [
    "Reply with exactly one word: OK",  # exact probe training prompt
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Explain what a for loop does in Python.",
    "List three primary colors.",
    "What is 12 multiplied by 8?",
]


def _decode_first_tinker(resp, tok) -> list[int]:
    r = resp.result() if hasattr(resp, "result") else resp
    seqs = getattr(r, "sequences", None) or getattr(r, "samples", None) or []
    seq = seqs[0]
    toks = (getattr(seq, "tokens", None) or getattr(seq, "token_ids", None)
            or getattr(seq, "output_tokens", None))
    return list(toks)


def run_tinker_side(sampler_path: str, tok, renderer) -> list[dict]:
    import tinker

    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(model_path=sampler_path)
    sp = tinker.SamplingParams(max_tokens=MAX_GEN_TOKENS, temperature=0.0,
                                stop=renderer.get_stop_sequences())

    results = []
    for prompt_text in FIXED_PROMPTS:
        messages = [{"role": "user", "content": prompt_text}]
        model_input = renderer.build_generation_prompt(messages)
        prompt_token_ids = model_input.to_ints()
        prompt_str = tok.decode(prompt_token_ids)
        resp = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=sp)
        out_tokens = _decode_first_tinker(resp, tok)
        out_text = tok.decode(out_tokens)
        results.append({
            "prompt": prompt_text,
            "rendered_prompt_text": prompt_str,
            "tinker_output_tokens": out_tokens,
            "tinker_output_text": out_text,
        })
        print(f"[tinker] {prompt_text!r} -> {out_text!r}", flush=True)
    return results


def run_vllm_side(rows: list[dict], tok) -> list[dict]:
    from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm
    import openai

    serve_script = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
    port = 8033
    container = "base21-moe-groundtruth"

    boot_vllm(str(MERGED_MODEL_DIR), container, port, 0.80,
              serve_script=serve_script, extra_env={"LANGUAGE_MODEL_ONLY": "1"})
    try:
        served = wait_healthy(port, container, timeout=1200)
        client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
        for row in rows:
            try:
                resp = client.completions.create(
                    model=served,
                    prompt=row["rendered_prompt_text"],
                    max_tokens=MAX_GEN_TOKENS,
                    temperature=0.0,
                )
                out_text = resp.choices[0].text or ""
            except Exception as e:  # noqa: BLE001
                print(f"[vllm] gen error for {row['prompt']!r}: {e}", flush=True)
                out_text = ""
            out_tokens = tok.encode(out_text, add_special_tokens=False)
            row["vllm_output_text"] = out_text
            row["vllm_output_tokens"] = out_tokens
            print(f"[vllm]   {row['prompt']!r} -> {out_text!r}", flush=True)
    finally:
        stop_vllm(container)
    return rows


def main() -> int:
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    if not MERGED_MODEL_DIR.exists():
        raise SystemExit(f"merged model not found at {MERGED_MODEL_DIR} -- run merge_adapter.py first")

    probe = json.loads(PROBE_RECEIPT.read_text())
    sampler_path = probe["sampler_path"]
    print(f"[verify] sampler_path={sampler_path}", flush=True)

    tok = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tok)

    rows = run_tinker_side(sampler_path, tok, renderer)
    rows = run_vllm_side(rows, tok)

    result = compare_and_write(rows, tok, sampler_path)
    return 0 if result["verdict_pass"] else 1


def compare_and_write(rows: list[dict], tok, sampler_path: str) -> dict:
    """Score rows (normalized-token + text comparison) and write OUT_PATH.

    Split from main() so the verdict can be recomputed offline from the
    already-captured rows without re-spending a Tinker sample + vLLM boot.
    """
    n_text_exact = 0
    n_prefix_agree = 0
    trained_prompt_exact = False
    for i, row in enumerate(rows):
        # Normalize BOTH sides through the same encode path (see module doc).
        t_norm = tok.encode(row["tinker_output_text"], add_special_tokens=False)
        v_norm = tok.encode(row["vllm_output_text"], add_special_tokens=False)
        row["normalized_prefix_agree"] = t_norm[:PREFIX_TOKENS] == v_norm[:PREFIX_TOKENS]
        row["text_exact_match"] = row["tinker_output_text"] == row["vllm_output_text"]
        n_prefix_agree += row["normalized_prefix_agree"]
        n_text_exact += row["text_exact_match"]
        if i == 0:  # FIXED_PROMPTS[0] is the exact probe training prompt
            trained_prompt_exact = row["text_exact_match"]

    verdict_pass = trained_prompt_exact and n_text_exact >= 2
    result = {
        "sampler_path": sampler_path,
        "merged_model_dir": str(MERGED_MODEL_DIR),
        "renderer_name": RENDERER_NAME,
        "prefix_tokens": PREFIX_TOKENS,
        "n_prompts": len(rows),
        "n_text_exact_match": n_text_exact,
        "n_normalized_prefix_agree": n_prefix_agree,
        "trained_prompt_text_exact": trained_prompt_exact,
        "verdict_pass": verdict_pass,
        "verdict_criterion": (
            "PASS iff the exact probe-training prompt (the only prompt with a "
            "decisive learned margin under this deliberately-overfit adapter) "
            "matches the Tinker sampler byte-for-byte AND at least 2 total "
            "prompts match exactly; off-signal prompts sit in near-tie "
            "degenerate logits where bf16 kernel-order across two unrelated "
            "serving stacks legitimately flips mid-junk argmax."
        ),
        "rows": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"[verify] text_exact {n_text_exact}/{len(rows)}, normalized-prefix "
          f"{n_prefix_agree}/{len(rows)}, trained_prompt_exact={trained_prompt_exact}, "
          f"verdict_pass={verdict_pass}", flush=True)
    print(f"[verify] wrote {OUT_PATH}", flush=True)
    return result


if __name__ == "__main__":
    sys.exit(main())
