"""Phase 0.3 Path B fallback — eval_judge via Unsloth FastLanguageModel.

Use only if Path A (vLLM-served LoRA) also fails to bind the MoE expert LoRA
declared via `target_parameters` in the 30/70 adapter config.

Unsloth's FastLanguageModel.from_pretrained() handles the MoE expert-LoRA
binding that raw PEFT 0.18.1 cannot. Trade-off: no batching kernels, so this
runs ~5-10× slower than vLLM. Acceptable for the 145-example seed-GT eval set.

Reuses eval/eval_judge.py for prompt extraction + GT parsing + Spearman math;
replaces the OpenAI HTTP client with direct in-process FastLanguageModel
generation.

Usage (inside unsloth-headless container after force-reinstalling pins):
    python -m scripts.eval_judge_unsloth \
      --base-path models/Qwen3-30B-A3B \
      --adapter-path adapters/qwen3-30b-wp-30_70 \
      --test-jsonl output/diagnostic/seeds_as_judge_test.jsonl \
      --output output/diagnostic/judge_30_70_seed_spearman.json

Status: STUB. Implementation deferred until Path A (vLLM) is tested. If Path A
binds the expert LoRA correctly we won't need this; if it doesn't, expand this
to mirror eval/eval_judge.py's main() with FastLanguageModel substitution.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(
        "ERROR: scripts/eval_judge_unsloth.py is a stub. Test Path A (vLLM-served "
        "LoRA via recipes/qwen3-30b-wp-30_70-vllm.yaml) first per "
        "output/diagnostic/baseline.md Phase 0.3 design. If vLLM also fails to "
        "bind the MoE expert LoRA, complete this script by porting eval/eval_judge.py's "
        "main() loop with the OpenAI HTTP client replaced by FastLanguageModel.from_pretrained() "
        "+ tokenizer.batch_decode() in-process generation.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
