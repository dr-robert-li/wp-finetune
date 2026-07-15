"""Bounded GB10 tooling smoke test for the v4 judge (GATE4-02, Plan 22-02).

Proves the Plan 22-01 sieve_arch adaptation is correct on the REAL v4 judge with a
single, bounded GB10 load -- the empirical half of GATE4-02. One serve-free
transformers forward pass over a tiny stimulus sample (default N=32,
max_seq_len=1024) captures the router (mlp.gate) output on
models/Qwen3.6-35B-A3B-judge-v4-s1-merged and writes a receipt asserting:

  - 40 hooks registered via sieve_arch.resolve_moe_layers (SC1)
  - per-layer strata (30 deltanet + 10 attention, at the config.layer_types
    indices) via sieve_arch.layer_strata (SC2)
  - router_logits last dim == 256 == config.num_experts, with a SEPARATE
    mlp.shared_expert module confirmed present -- the empirical proof the
    shared expert never appears in router_logits (SC3)
  - the traversal root that ACTUALLY resolved (SC4 / closes the ROADMAP-SC1-
    vs-20-04 open question)

NOT the ~6h30m full profiling pass -- that is Phase 25's job.

Usage:
    python -m scripts.sieve_v4_tooling_smoke
    python -m scripts.sieve_v4_tooling_smoke --allow-cpu   # NOT recommended
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from scripts import sieve_arch  # noqa: E402

DEFAULT_MODEL_PATH = "models/Qwen3.6-35B-A3B-judge-v4-s1-merged"
DEFAULT_DATA_PATH = "data/reasoning_dataset/openai_val.jsonl"
DEFAULT_OUTPUT = "output/sieve-v4/tooling_smoke.json"
N_EXAMPLES = 32
MAX_SEQ_LEN = 1024
EXPECTED_ATTENTION_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]


def _resolved_traversal_root(model) -> str | None:
    """Walk the same candidate roots sieve_arch.resolve_moe_layers tries, and
    return the first one that resolves to a non-empty layer list -- for the
    receipt only (resolve_moe_layers itself remains the single source of truth
    for hook registration; this is a read-only diagnostic mirror of its walk).
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for path in sieve_arch._MOE_LAYER_ROOT_CANDIDATES:
        obj = base
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and len(list(obj)) > 0:
            return path
    return None


def _load_examples(data_path: Path, n_examples: int) -> list[dict]:
    examples = []
    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(json.loads(line))
            if len(examples) >= n_examples:
                break
    return examples


def run_smoke(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    n_examples: int = N_EXAMPLES,
    max_seq_len: int = MAX_SEQ_LEN,
) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    result: dict = {
        "model_path": str(model_path),
        "n_examples": n_examples,
        "max_seq_len": max_seq_len,
    }

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    print(f"Loading model from {model_path} (bf16, device_map=auto) ...")
    # Deviation (Rule 1 - bug): the plan text names AutoModelForCausalLM, but this
    # checkpoint's config.architectures is Qwen3_5MoeForConditionalGeneration (a VL
    # composite -- vision tower weights are present on disk) and text weights are
    # saved under the nested `model.language_model.layers.*` key convention.
    # AutoModelForCausalLM resolves qwen3_5_moe -> Qwen3_5MoeForCausalLM, whose FLAT
    # `model.layers.*` state_dict has ZERO overlap with the checkpoint's text keys
    # (692/693 keys missing, confirmed via a meta-device key-set diff before this
    # load) -- that would silently produce a randomly-initialized forward pass, not
    # a real one. AutoModelForImageTextToText resolves Qwen3_5MoeForConditionalGeneration,
    # whose state_dict has 0 missing keys against the checkpoint (only 19 unrelated
    # `mtp.*` multi-token-prediction keys are unexpected/ignored). Using the class
    # that actually matches every text weight is required for GATE4-02's empirical
    # claim to be true.
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    hooks = []
    try:
        # --- SC1: hooks + traversal root ---
        n_layers, n_experts = sieve_arch.arch_dims(model.config)
        result["expected_n_layers"] = n_layers
        result["config_num_experts"] = n_experts
        result["resolved_traversal_root"] = _resolved_traversal_root(model)

        moe_layers = sieve_arch.resolve_moe_layers(model)
        result["hooks_registered"] = len(moe_layers)

        # --- SC3 setup: shared-expert module presence (structural, no forward needed) ---
        result["shared_expert_module_present"] = any(
            getattr(mlp, "shared_expert", None) is not None for _, mlp in moe_layers
        )

        capture: dict = {"last_dim": None}

        def make_probe_hook():
            def hook(module, inputs, outputs):
                if capture["last_dim"] is None:
                    tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                    capture["last_dim"] = int(tensor.shape[-1])
            return hook

        for _layer_idx, mlp in moe_layers:
            hooks.append(mlp.gate.register_forward_hook(make_probe_hook()))

        # --- SC2: strata ---
        strata = sieve_arch.layer_strata(model.config)
        result["strata_counts"] = {
            "deltanet": strata.count(sieve_arch.DELTANET_STRATUM),
            "attention": strata.count(sieve_arch.ATTENTION_STRATUM),
        }
        result["attention_layer_indices"] = [
            i for i, s in enumerate(strata) if s == sieve_arch.ATTENTION_STRATUM
        ]

        # --- Bounded forward pass ---
        examples = _load_examples(data_path, n_examples)
        print(f"Running bounded forward pass: {len(examples)} examples, max_seq_len={max_seq_len}")
        with torch.no_grad():
            for ex in examples:
                messages = ex.get("messages", [])
                if messages:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                else:
                    text = ex.get("text", "")
                enc = tokenizer(
                    text, return_tensors="pt", max_length=max_seq_len, truncation=True
                )
                model(input_ids=enc["input_ids"].to(model.device))

        # --- SC3: empirical router_logits shape ---
        result["router_logits_last_dim"] = capture["last_dim"]
        result["shared_expert_in_router_logits"] = (
            capture["last_dim"] != n_experts if capture["last_dim"] is not None else None
        )

        status = (
            result["hooks_registered"] == 40 == result["expected_n_layers"]
            and result["router_logits_last_dim"] == 256 == result["config_num_experts"]
            and result["shared_expert_in_router_logits"] is False
            and result["shared_expert_module_present"] is True
            and result["strata_counts"] == {"deltanet": 30, "attention": 10}
            and result["attention_layer_indices"] == EXPECTED_ATTENTION_INDICES
            and isinstance(result["resolved_traversal_root"], str)
            and bool(result["resolved_traversal_root"])
        )
        result["status"] = "pass" if status else "fail"
    finally:
        for h in hooks:
            h.remove()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(
                f"Post-cleanup CUDA memory allocated: "
                f"{torch.cuda.memory_allocated() / (1024**3):.2f} GiB"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote receipt: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Bounded GB10 tooling smoke for the v4 judge (GATE4-02)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution (NOT recommended for a 35B model)",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        print(
            "ERROR: torch.cuda.is_available() is False -- refusing to run a 35B forward "
            "pass on CPU. Pass --allow-cpu to override (NOT recommended)."
        )
        sys.exit(2)

    result = run_smoke(
        model_path=PROJECT_ROOT / args.model_path,
        data_path=PROJECT_ROOT / args.data_path,
        output_path=PROJECT_ROOT / args.output,
        n_examples=args.n_examples,
        max_seq_len=args.max_seq_len,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
