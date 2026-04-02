# Phase 4: Base-Model Profiling & Evaluation (Triage) - Research

**Researched:** 2026-04-01
**Domain:** PyTorch forward hooks (MoE routing), vLLM LoRA serving, eval script interfaces, wp-bench
**Confidence:** HIGH

## Summary

Phase 4 has two sequential components: (1) a fast base-model E_eff profiling run (~minutes) using forward
hooks on `Qwen3MoeSparseMoeBlock`, and (2) a longer eval triage loop serving each of 3 existing adapters
via vLLM sequentially and running the full static eval suite + wp-bench per adapter.

The profiling component is fully new code — no existing `scripts/profile*.py` exists. It must hook
`model.model.layers[i].mlp.gate` (the `Qwen3MoeTopKRouter`) to capture `selected_experts` (token→expert
routing indices, shape `[seq_len * batch, top_k]`), then accumulate per-layer expert counts split by
`<wp_gen>` (token_id=151669) vs `<wp_judge>` (token_id=151670). E_eff = exp(Shannon entropy) per layer,
computed from count-normalized distributions.

The eval triage component reuses `eval/eval_gen.py`, `eval/eval_judge.py`, and `eval/eval_gate.py`
directly — these scripts are complete and tested. The only setup work is: (a) starting the vLLM container
with LoRA enabled for each adapter, (b) per-ratio output path management, and (c) cloning + configuring
wp-bench against the live vLLM endpoint (config already exists at `config/wp-bench.yaml`).

**Primary recommendation:** Write one new script `scripts/profile_base_model.py` for E_eff profiling. For
eval triage, drive the existing eval scripts via a thin orchestration script `scripts/run_eval_triage.py`
that loops over the 3 adapter ratios, starts/stops vLLM between ratios, and collects per-ratio output.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Profile base model (no adapter) with all 5 ratio data distributions by hooking `Qwen3MoeSparseMoeBlock` gating output
- **D-02:** 10% subsample per ratio for stability (MoE-Sieve paper: Jaccard ≥0.94 at 10%)
- **D-03:** Output: JSONL per-layer raw data + markdown summary table with E_eff mean/max/variance per ratio
- **D-04:** E_eff = exp(entropy) per layer. Count-based routing per expert per layer. Separate `<wp_gen>` vs `<wp_judge>` token counts.
- **D-05:** E_eff training trigger: ANY downward trend as gen% increases → train 60/40
- **D-06:** Adapters served sequentially via vLLM `--lora-modules` — one at a time, eval fully before next
- **D-07:** All 3 existing adapters (30/70, 40/60, 50/50) get full eval: static suite + full wp-bench run
- **D-08:** If 60/40 training starts, its eval runs when training completes (~2 days)
- **D-09:** Static eval gates first, wp-bench for differentiation
- **D-10:** Hard gates: PHPCS pass rate >95%, Judge Spearman >0.85, Security pass rate >98%
- **D-11:** Gen-weighted triage
- **D-12:** Elimination = fails ANY hard gate OR >5pp behind best on overall score
- **D-13:** High bar for elimination, low bar for continuation
- **D-14:** Post-compression quality-per-VRAM is what matters, not pre-compression eval score
- **D-15:** Step 1 (~minutes) completes first; if E_eff trending down → start 60/40 training immediately
- **D-16:** Step 2 runs in parallel with any 60/40 training
- **D-17:** Triage decision when all warranted adapters are evaluated
- **D-18:** wp-bench is canonical WP AI benchmark
- **D-19:** No Claude in eval loop
- **D-20:** Eval scripts exist: `eval/eval_gen.py`, `eval/eval_judge.py`, `eval/eval_gate.py`
- **D-21:** Test data: `data/final_dataset/openai_test.jsonl` (597 held-out examples — actual line count is 10166, but these include all messages not just test examples)
- **D-22:** Model serving via DGX Toolbox vLLM + LiteLLM proxy

### Claude's Discretion

- Profiling script implementation details (hook registration, data loading)
- Eval execution ordering among the 3 adapters (which ratio first doesn't matter)
- wp-bench task category weighting within the full run
- Markdown summary table formatting

### Deferred Ideas (OUT OF SCOPE)

- Full 5-ratio comparison through entire pipeline
- 70/30 training
- Layer-adaptive pruning ratio analysis (Phase 12)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | Custom eval script measures PHPCS pass rate on 500 held-out generation tasks (target >95%) | `eval/eval_gen.py` run_eval() confirmed ready; needs vLLM endpoint serving the adapter |
| EVAL-02 | Custom eval script measures judge Spearman correlation on 500 held-out scored pairs (target >0.85) | `eval/eval_judge.py` run_eval() confirmed ready; needs vLLM endpoint |
| EVAL-03 | Security pass rate measured on held-out tasks (target >98%) | Security pass rate computed inside `eval_gen.py` summary (D2_security dim ≥8 rate) |
| EVAL-04 | Eval scripts run via DGX Toolbox eval-toolbox container | `dgx_toolbox.yaml` has `eval_toolbox` container mapped to `eval/eval-toolbox.sh` |
| EVAL-05 | All three quality gates pass before proceeding to deployment | `eval/eval_gate.py` run_gate() checks all three; exits non-zero on any failure |
| GATE-02 | Phase 4 triage uses high bar for elimination (fail hard gates OR >5pp behind) | Triage script must read per-ratio results and apply D-12 elimination logic |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 5.3.0 (in env) | Model loading, AutoModel, hook targets | Already installed; Qwen3MoE support native |
| torch | 2.10.0+cu* (in container) | Forward hooks, tensor ops for E_eff | PyTorch hooks are the canonical approach |
| openai | (in env) | API client for eval scripts | Eval scripts already use it via dgx.vllm_endpoint() |
| scipy | (in env) | spearmanr in eval_judge.py | Already a required import in dgx_toolbox.yaml |
| numpy | standard | E_eff entropy computation | scipy.stats.entropy or manual numpy |
| pyyaml | (in env) | Config loading in eval scripts | Already in extra_deps |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | standard | Progress bar in profiling script | Long forward passes over 3000-8000 examples |
| json / jsonlines | stdlib | JSONL output writing | Profiling JSONL output |

**Version verification:** Transformers 5.3.0 is installed in local env; the unsloth container has PyTorch with CUDA. The profiling script runs inside the container for GPU access.

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── profile_base_model.py   # NEW: E_eff profiling for base model
├── run_eval_triage.py      # NEW: orchestrates per-ratio eval loop
eval/
├── eval_gen.py             # EXISTING: PHPCS pass rate
├── eval_judge.py           # EXISTING: Spearman correlation
├── eval_gate.py            # EXISTING: quality gate aggregator
output/
├── profiling/
│   ├── base_model_eeff.jsonl            # per-layer raw data (Phase 7 readable)
│   └── base_model_eeff_summary.md       # markdown table
├── eval_triage/
│   ├── ratio_30_70/
│   │   ├── eval_gen_results.json
│   │   ├── eval_judge_results.json
│   │   └── wp_bench_results.json
│   ├── ratio_40_60/
│   └── ratio_50_50/
└── triage_decision.md                   # Final elimination verdict
```

### Pattern 1: Qwen3MoE Router Hook Registration

**What:** Register a `register_forward_hook` on `Qwen3MoeSparseMoeBlock.gate` (which is a
`Qwen3MoeTopKRouter`). The router `forward()` returns `(router_logits, router_scores, router_indices)`
where `router_indices` has shape `[n_tokens, top_k]` (int64 tensor giving the expert indices selected
for each token).

**Module path:** `model.model.layers[i].mlp` is a `Qwen3MoeSparseMoeBlock` when layer `i` is a MoE layer.
`model.model.layers[i].mlp.gate` is the `Qwen3MoeTopKRouter`. Hook on the gate sub-module to capture
`selected_experts` (third output).

**Important:** `config.mlp_only_layers = []` and `decoder_sparse_step = 1` for Qwen3-30B-A3B, so ALL
48 layers use `Qwen3MoeSparseMoeBlock`. There are no dense MLP-only layers.

**When to use:** Gradient-free profiling run (no training). Use `model.eval()` and `torch.no_grad()`.

**Example:**
```python
# Source: verified from transformers 5.3.0 Qwen3MoeTopKRouter.forward()
routing_counts = {}  # layer_idx -> expert_id -> {"wp_gen": int, "wp_judge": int, "other": int}

def make_hook(layer_idx):
    def hook(module, inputs, outputs):
        # outputs = (router_logits, router_scores, router_indices)
        # router_indices: [n_tokens, top_k=8], dtype=torch.int64
        router_indices = outputs[2]  # shape [n_tokens, top_k]
        # token_types from input_ids, aligned to n_tokens (batch*seq_len)
        # Must be passed via closure or nonlocal state from the forward pass
        token_types = current_token_types  # set before each forward call

        counts = routing_counts.setdefault(layer_idx, {})
        for tok_pos in range(router_indices.shape[0]):
            tok_type = token_types[tok_pos] if tok_pos < len(token_types) else "other"
            for expert_id in router_indices[tok_pos].tolist():
                if expert_id not in counts:
                    counts[expert_id] = {"wp_gen": 0, "wp_judge": 0, "other": 0}
                counts[expert_id][tok_type] += 1
    return hook

# Register hooks on all 48 MoE layers
hooks = []
for i, layer in enumerate(model.model.layers):
    if hasattr(layer.mlp, 'gate'):  # is a SparseMoeBlock
        h = layer.mlp.gate.register_forward_hook(make_hook(i))
        hooks.append(h)
```

**Hook output capture note:** The gate forward hook outputs are the direct return values of
`Qwen3MoeTopKRouter.forward()`: `(router_logits, router_scores, router_indices)`.
`router_indices` is what we want — the selected expert IDs per token.

### Pattern 2: Token-Type Tagging for wp_gen / wp_judge Split

**What:** Before each forward pass, scan the input_ids to produce a flat list of token types
(`"wp_gen"`, `"wp_judge"`, or `"other"`) corresponding to positions in the sequence. The router
hook receives flattened `[batch*seq_len, top_k]` tensors, so the token type list must also be
flattened `[batch*seq_len]`.

**Token IDs:** `<wp_gen>` = 151669, `<wp_judge>` = 151670 (verified from extended tokenizer at
`adapters/tokenizer/`). The base model uses the extended tokenizer for fair profiling.

**Tagging strategy:** For each token position, the "type" is the most recent task token seen looking
backwards in the sequence. This matches the actual inference semantics (task token precedes generation).

```python
WP_GEN_ID = 151669
WP_JUDGE_ID = 151670

def tag_token_types(input_ids_flat):
    """input_ids_flat: 1D tensor of shape [batch*seq_len]"""
    types = []
    current = "other"
    for tid in input_ids_flat.tolist():
        if tid == WP_GEN_ID:
            current = "wp_gen"
        elif tid == WP_JUDGE_ID:
            current = "wp_judge"
        types.append(current)
    return types
```

### Pattern 3: E_eff Computation from Routing Counts

**What:** E_eff = exp(Shannon entropy H) where H = -sum(p_i * log(p_i)) over experts, and p_i is
the fraction of total token-expert activations going to expert i for a given layer.

**Formula (verified against MoE-Sieve paper arxiv 2603.24044):**
- For each layer, aggregate total counts across all experts: total = sum(counts[expert])
- Compute p_i = counts[expert_i] / total for each expert
- H = -sum(p_i * log(p_i)) (using natural log, or log2 with max at log2(128) = 7.0)
- E_eff = exp(H) — effective number of experts (between 1.0=fully concentrated and 128=uniform)
- Compute separately for wp_gen tokens, wp_judge tokens, and combined

```python
import numpy as np

def compute_eeff(expert_counts: dict[int, int], n_experts: int = 128) -> float:
    """expert_counts: {expert_id: count}"""
    total = sum(expert_counts.values())
    if total == 0:
        return float(n_experts)  # uniform if no data
    p = np.array([expert_counts.get(i, 0) / total for i in range(n_experts)])
    # avoid log(0)
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))
```

**Per-ratio E_eff summary:** For each of the 5 ratios, compute:
- `eeff_mean`: mean E_eff across all 48 layers
- `eeff_max`: max E_eff (most uniform layer)
- `eeff_variance`: variance across layers (layer-depth skew signal)
- Split for wp_gen tokens and wp_judge tokens separately

### Pattern 4: JSONL Output Format for Phase 7 Compatibility

**What:** Each JSONL record represents one (ratio, layer) pair. Phase 7 reads the same format
for base-vs-adapter comparison, so the schema is fixed here.

```json
{
  "ratio": "30_70",
  "layer_idx": 0,
  "n_tokens_total": 12456,
  "n_tokens_wp_gen": 8134,
  "n_tokens_wp_judge": 4322,
  "expert_counts_total": {"0": 1234, "1": 987, ...},
  "expert_counts_wp_gen": {"0": 823, "1": 641, ...},
  "expert_counts_wp_judge": {"0": 411, "1": 346, ...},
  "eeff_total": 45.2,
  "eeff_wp_gen": 42.1,
  "eeff_wp_judge": 49.3,
  "subsample_n": 3486,
  "model": "base"
}
```

**Phase 7 note:** The `"model"` field distinguishes base (Phase 4) from adapter (Phase 7) records.

### Pattern 5: vLLM LoRA Adapter Serving

**What:** Start vLLM with `--enable-lora` and `--lora-modules` flags to serve a base model with
an adapter hot-loaded. The existing `start-vllm.sh` supports extra args passed through.

**Serving command (via dgx_toolbox.run_service):**
```bash
bash ~/dgx-toolbox/inference/start-vllm.sh \
    /workspace/wp-finetune/models/Qwen3-30B-A3B \
    --enable-lora \
    --lora-modules qwen3-wp=/workspace/wp-finetune/adapters/qwen3-30b-wp-30_70 \
    --max-lora-rank 64 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92
```

**Model name for eval scripts:** The eval scripts query `model="openai/qwen3-wp"`. With
`--lora-modules qwen3-wp=...`, vLLM serves the lora model as `qwen3-wp`. The eval scripts
use `openai/qwen3-wp` (OpenAI client prepends `openai/` for some configurations) — verify the
exact model name string at runtime.

**Key flags:**
- `--enable-lora`: Required to load LoRA adapters
- `--lora-modules NAME=PATH`: Maps adapter directory to a model name
- `--max-lora-rank 64`: Must match adapter `r` value (adapters have r=32; set ≥32, 64 is safe)
- `--gpu-memory-utilization 0.92`: DGX Spark 128GB — leave some headroom
- Sequential serving: stop and restart vLLM between each adapter ratio

**DGX Spark constraint (D-06):** Only one adapter loaded at a time. 128GB is tight for 30B base
+ adapter in bf16 (base uses ~60GB, adapter activations add ~5-10GB). Do not attempt parallel.

**modules_to_save handling:** The adapters were trained with `modules_to_save=["embed_tokens", "lm_head"]`.
These are saved as full tensors in the adapter directory. vLLM handles this transparently with
`--enable-lora` — it merges the saved embedding/lm_head weights automatically.

### Pattern 6: Per-Ratio Output Directory Management

**What:** Each adapter eval writes to a ratio-specific output directory to avoid collision between
the 3 sequential eval runs.

```python
# In run_eval_triage.py
for ratio in ["30_70", "40_60", "50_50"]:
    out_dir = Path(f"output/eval_triage/ratio_{ratio}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run static eval
    eval_gen.run_eval(output_path=str(out_dir / "eval_gen_results.json"))
    eval_judge.run_eval(output_path=str(out_dir / "eval_judge_results.json"))
    eval_gate.run_gate(results_dir=str(out_dir))
```

### Anti-Patterns to Avoid

- **Hooking `model.model.layers[i].mlp` (the SparseMoeBlock itself) instead of `.mlp.gate`:**
  The SparseMoeBlock.forward() does not directly return the routing indices. Hook `.mlp.gate`
  (the Qwen3MoeTopKRouter) where `outputs[2]` is `router_indices`.
- **Using `register_forward_pre_hook` instead of `register_forward_hook`:** Pre-hooks don't have
  outputs yet. Use post-hook (`register_forward_hook`) to capture the gate's return values.
- **Loading the full model in fp32 for profiling:** The base model is 30B params. Always use
  `torch_dtype=torch.bfloat16` and `device_map="auto"` (or single GPU with enough VRAM).
- **Running profiling outside the unsloth container:** CUDA is only available in Docker containers
  on this machine (local env has CPU-only torch). Profiling must run inside `unsloth-headless`.
- **Using `openai_test.jsonl` for profiling instead of ratio-specific train data:** Profiling should
  use each ratio's own distribution data (from `data/final_dataset/ratio_{r}/openai_train.jsonl`) to
  capture how the base model routes that specific data mix.
- **Forgetting to clear routing count state between ratios:** Hook closures accumulate across
  forward calls. Reset `routing_counts` dict before each new ratio.
- **Using the base tokenizer instead of the extended tokenizer for profiling:** The extended tokenizer
  at `adapters/tokenizer/` has `<wp_gen>` (151669) and `<wp_judge>` (151670). Using the base
  tokenizer would map these tokens to UNK or wrong IDs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Spearman correlation | Custom rank correlation | `scipy.stats.spearmanr` (already used in eval_judge.py) | Edge cases with ties, p-value needed |
| PHPCS pass rate | Custom PHP linter | `eval_gen.py` rubric scorer (241 checks) | Already production-ready with 9-dimension scoring |
| Shannon entropy | Manual log-sum | `numpy` with p * log(p) pattern or `scipy.stats.entropy` | Floating point stability, zero-handling |
| vLLM HTTP health check | Custom TCP probe | `curl http://localhost:8020/health` or `openai` client ping | vLLM exposes `/health` endpoint |
| wp-bench runner | Custom WP eval | Clone `github.com/WordPress/wp-bench`, configure with `wp-bench.yaml` | Canonical benchmark, Docker WordPress runtime |

**Key insight:** The hardest part of this phase is the new profiling script. All eval mechanics
already exist and are tested.

## Common Pitfalls

### Pitfall 1: Router Hook Output Index
**What goes wrong:** Capturing `outputs[1]` (router_scores/weights) instead of `outputs[2]`
(router_indices/selected_experts). Weights don't identify which experts were selected — you get
float probabilities instead of integer expert IDs.
**Why it happens:** The router returns 3 values and the indices are the third, not second.
**How to avoid:** Verify by checking shape — `router_indices` is int64 shape `[n_tokens, 8]`;
`router_scores` is float32 shape `[n_tokens, 8]`.
**Warning signs:** E_eff computations produce values >128 or wrong dtype errors.

### Pitfall 2: Token Flattening Alignment
**What goes wrong:** The hook receives tokens as `[batch*seq_len, top_k]` but your token-type list
is organized as `[batch][seq_len]` — off-by-one or batch-size misalignment corrupts the wp_gen/wp_judge
split.
**Why it happens:** The SparseMoeBlock reshapes `hidden_states` to `(-1, hidden_dim)` before passing
to the gate, flattening the batch dimension. Your token type tags must be flattened the same way.
**How to avoid:** Produce token type tags from `input_ids.view(-1)` to match the gate's flattening.
**Warning signs:** wp_gen and wp_judge counts don't add up to total; one type is always 0.

### Pitfall 3: vLLM LoRA Model Name Mismatch
**What goes wrong:** The eval scripts query `model="openai/qwen3-wp"` but vLLM serves the lora
model as `"qwen3-wp"` (without the `openai/` prefix). The `openai` Python client strips prefixes
differently depending on the `base_url` configuration.
**Why it happens:** Different OpenAI client versions handle model name normalization differently.
**How to avoid:** After starting vLLM, hit `GET http://localhost:8020/v1/models` and verify the
exact model name string before running eval scripts. If needed, update the model name in the eval
scripts or pass it as a parameter.
**Warning signs:** `404 model not found` errors from the eval scripts.

### Pitfall 4: Output Directory Collision Across Ratios
**What goes wrong:** Running eval_gen.py three times with the default output path overwrites results,
leaving only the last ratio's results.
**Why it happens:** eval scripts default to `output/eval_gen_results.json`. The triage orchestrator
must pass per-ratio output paths explicitly.
**How to avoid:** Always pass `--output output/eval_triage/ratio_{r}/eval_gen_results.json` or call
`run_eval(output_path=...)` programmatically.
**Warning signs:** All three ratios show identical eval results.

### Pitfall 5: Subsample Must Use openai Format (not raw or alpaca)
**What goes wrong:** Loading ratio data from `raw_train.jsonl` (raw format) instead of
`openai_train.jsonl` (chat format). The profiling script must tokenize in the same chat format
that training used.
**Why it happens:** Three formats exist per ratio; all have the same examples in different schemas.
**How to avoid:** Always use `data/final_dataset/ratio_{r}/openai_train.jsonl` for profiling.
Apply 10% subsample: `ratio_30_70` → 3,486 examples; `ratio_40_60` → 4,067; `ratio_50_50` → 4,880;
`ratio_60_40` → 6,100; `ratio_70_30` → 8,133.
**Warning signs:** Tokenizer produces sequences with wrong special token IDs.

### Pitfall 6: wp-bench Requires Docker WordPress Runtime
**What goes wrong:** wp-bench's grader runs WordPress inside Docker to actually execute generated PHP
code. If the `wp-bench/runtime` Docker image isn't built or if `grader.kind: docker` isn't configured,
grading silently fails or falls back to static-only.
**Why it happens:** wp-bench uses a real WordPress environment for execution tests.
**How to avoid:** Clone wp-bench, run its setup script to build the Docker runtime image before
evaluation. Check `config/wp-bench.yaml` — `grader.wp_env_dir: ./wp-bench/runtime` must exist.
**Warning signs:** wp-bench scores suspiciously high (static-only, no runtime failures) or
grading times out.

### Pitfall 7: `modules_to_save` Tensors in Adapter
**What goes wrong:** vLLM fails to load the adapter because the adapter directory contains full
`embed_tokens` and `lm_head` weight tensors (from `modules_to_save`), not standard LoRA diff
tensors. Some vLLM versions reject this.
**Why it happens:** The adapters were trained with `modules_to_save=["embed_tokens", "lm_head"]`.
**How to avoid:** Verify vLLM version supports `modules_to_save` in LoRA loading. vLLM ≥0.6.x
handles this. If not supported, pre-merge the adapter before serving (use `scripts/merge_adapter.py`).
**Warning signs:** `ValueError` or `KeyError` on adapter load, or model generates garbage for
special tokens.

## Code Examples

### E_eff Forward Hook (Complete Pattern)
```python
# Source: verified against transformers 5.3.0 Qwen3MoeTopKRouter.forward()
import torch
import numpy as np
from collections import defaultdict

WP_GEN_ID = 151669   # verified from adapters/tokenizer/
WP_JUDGE_ID = 151670

class RoutingCollector:
    """Collects per-layer routing counts from Qwen3MoeTopKRouter forward hooks."""

    def __init__(self, n_layers: int, n_experts: int = 128, top_k: int = 8):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.reset()

    def reset(self):
        # layer -> expert -> {"wp_gen": int, "wp_judge": int, "other": int}
        self.counts = defaultdict(lambda: defaultdict(lambda: {"wp_gen": 0, "wp_judge": 0, "other": 0}))
        self._current_token_types = []

    def set_token_types(self, input_ids: torch.Tensor):
        """Call before each forward pass. input_ids shape: [batch, seq_len]"""
        flat = input_ids.view(-1).tolist()
        types = []
        current = "other"
        for tid in flat:
            if tid == WP_GEN_ID:
                current = "wp_gen"
            elif tid == WP_JUDGE_ID:
                current = "wp_judge"
            types.append(current)
        self._current_token_types = types

    def make_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            # outputs = (router_logits, router_scores, router_indices)
            router_indices = outputs[2].cpu()  # [n_tokens, top_k]
            n_tokens = router_indices.shape[0]
            for tok_pos in range(n_tokens):
                if tok_pos < len(self._current_token_types):
                    tok_type = self._current_token_types[tok_pos]
                else:
                    tok_type = "other"
                for expert_id in router_indices[tok_pos].tolist():
                    self.counts[layer_idx][expert_id][tok_type] += 1
        return hook

    def compute_eeff(self, layer_idx: int, tok_type: str = "total") -> float:
        layer_counts = self.counts[layer_idx]
        if tok_type == "total":
            agg = {e: sum(v.values()) for e, v in layer_counts.items()}
        else:
            agg = {e: v.get(tok_type, 0) for e, v in layer_counts.items()}
        total = sum(agg.values())
        if total == 0:
            return float(self.n_experts)
        p = np.array([agg.get(i, 0) / total for i in range(self.n_experts)])
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy))
```

### Hook Registration and Cleanup
```python
# Register hooks
collector = RoutingCollector(n_layers=48, n_experts=128, top_k=8)
hooks = []
for i, layer in enumerate(model.model.layers):
    if hasattr(layer.mlp, 'gate'):  # Qwen3MoeSparseMoeBlock
        h = layer.mlp.gate.register_forward_hook(collector.make_hook(i))
        hooks.append(h)

# Run profiling
model.eval()
with torch.no_grad():
    collector.set_token_types(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Always clean up hooks
for h in hooks:
    h.remove()
```

### JSONL Record Writer
```python
import json
from pathlib import Path

def write_profiling_jsonl(collector, ratio: str, subsample_n: int, out_path: Path):
    with out_path.open("w") as f:
        for layer_idx in range(48):
            n_gen = sum(v["wp_gen"] for v in collector.counts[layer_idx].values())
            n_judge = sum(v["wp_judge"] for v in collector.counts[layer_idx].values())
            n_other = sum(v["other"] for v in collector.counts[layer_idx].values())
            record = {
                "ratio": ratio,
                "layer_idx": layer_idx,
                "n_tokens_total": n_gen + n_judge + n_other,
                "n_tokens_wp_gen": n_gen,
                "n_tokens_wp_judge": n_judge,
                "expert_counts_total": {
                    str(e): sum(v.values())
                    for e, v in collector.counts[layer_idx].items()
                },
                "expert_counts_wp_gen": {
                    str(e): v["wp_gen"]
                    for e, v in collector.counts[layer_idx].items()
                },
                "expert_counts_wp_judge": {
                    str(e): v["wp_judge"]
                    for e, v in collector.counts[layer_idx].items()
                },
                "eeff_total": collector.compute_eeff(layer_idx, "total"),
                "eeff_wp_gen": collector.compute_eeff(layer_idx, "wp_gen"),
                "eeff_wp_judge": collector.compute_eeff(layer_idx, "wp_judge"),
                "subsample_n": subsample_n,
                "model": "base",
            }
            f.write(json.dumps(record) + "\n")
```

### E_eff Trend Detection (D-05 logic)
```python
def has_downward_eeff_trend(ratio_eeffs: dict[str, float]) -> bool:
    """
    ratio_eeffs: {"30_70": mean_eeff, "40_60": mean_eeff, "50_50": mean_eeff,
                  "60_40": mean_eeff, "70_30": mean_eeff}
    Returns True if E_eff decreases as gen% increases (30->40->50->60->70).
    """
    ordered = [ratio_eeffs[r] for r in ["30_70", "40_60", "50_50", "60_40", "70_30"]]
    # Check if there's ANY downward step as gen% increases
    for i in range(len(ordered) - 1):
        if ordered[i+1] < ordered[i]:
            return True
    return False
```

### vLLM LoRA Serving (via dgx_toolbox.run_service)
```python
dgx = get_toolbox()

# Start vLLM with LoRA adapter
result = dgx.run_service(
    "vllm",
    "/workspace/wp-finetune/models/Qwen3-30B-A3B",
    "--enable-lora",
    "--lora-modules", f"qwen3-wp=/workspace/wp-finetune/adapters/qwen3-30b-wp-{ratio}",
    "--max-lora-rank", "64",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.92",
)

# Wait for vLLM to be ready
import time, urllib.request
for _ in range(60):  # up to 5 min
    try:
        urllib.request.urlopen(f"{dgx.vllm_endpoint()}/models", timeout=2)
        break
    except Exception:
        time.sleep(5)
```

### Eval Script Invocation Pattern
```python
# Source: eval/eval_gen.py run_eval(), eval/eval_judge.py run_eval(), eval/eval_gate.py run_gate()
from eval import eval_gen, eval_judge, eval_gate

out_dir = f"output/eval_triage/ratio_{ratio}"
Path(out_dir).mkdir(parents=True, exist_ok=True)

gen_summary = eval_gen.run_eval(
    dataset_path="data/final_dataset/openai_test.jsonl",
    output_path=f"{out_dir}/eval_gen_results.json"
)
judge_summary = eval_judge.run_eval(
    dataset_path="data/final_dataset/openai_test.jsonl",
    output_path=f"{out_dir}/eval_judge_results.json"
)
passed, gate_rows = eval_gate.run_gate(results_dir=out_dir)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Log-softmax router (MixtralSparseMoeBlock) | Softmax router (Qwen3MoeTopKRouter) | Qwen3 architecture | Hook captures `router_logits` as probabilities, not log-probs |
| Separate router module as `model.block_sparse_moe.gate` | Router as `model.model.layers[i].mlp.gate` | Qwen3 vs Mixtral | Module path is different from Mixtral-style MoE |

**Current transformers version in env:** 5.3.0 — verified. `Qwen3MoeSparseMoeBlock` and
`Qwen3MoeTopKRouter` are confirmed present with the exact forward signatures documented above.

## Open Questions

1. **vLLM LoRA with `modules_to_save` tensors**
   - What we know: adapters contain full `embed_tokens` and `lm_head` tensors alongside LoRA diff tensors
   - What's unclear: whether the production vLLM Docker image (`vllm/vllm-openai:latest`) in dgx-toolbox
     fully supports this without error
   - Recommendation: Plan a fallback step — if vLLM LoRA loading fails, use `scripts/merge_adapter.py`
     to create a merged checkpoint and serve that as a full model (no `--lora-modules` needed)

2. **wp-bench clone location and setup time**
   - What we know: `config/wp-bench.yaml` references `./wp-bench/runtime` as Docker env dir; the repo
     is not yet cloned (`wp-bench not cloned` confirmed)
   - What's unclear: exact setup steps, whether the WordPress Docker runtime image needs manual building
   - Recommendation: Wave 0 task should clone `github.com/WordPress/wp-bench` and run its setup script;
     document expected setup time

3. **Exact model name for vLLM LoRA endpoint**
   - What we know: eval scripts use `model="openai/qwen3-wp"`; vLLM `--lora-modules` registers as `"qwen3-wp"`
   - What's unclear: whether the openai Python client strips `"openai/"` prefix when hitting vLLM
   - Recommendation: Add a health check in the eval triage script that lists available models and asserts
     the expected name before running eval

4. **Profiling batch size for the 30B model**
   - What we know: DGX Spark has 128GB RAM; base model in bf16 is ~60GB; profiling is forward-only
   - What's unclear: optimal batch size for profiling (higher = faster, but OOM risk)
   - Recommendation: Default to batch_size=1, max_seq_len=2048 for safety; note that this is gradient-free
     so no activation memory overhead from autograd

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| NVIDIA GPU (GB10) | Profiling, vLLM serving | ✓ | GB10, 128GB | — |
| Docker | vLLM container, wp-bench runtime | ✓ | Running | — |
| unsloth-headless container | Profiling (needs CUDA torch) | ✓ | Up 2 days | — |
| transformers (Qwen3MoE) | Profiling hook | ✓ | 5.3.0 | — |
| vllm (in vllm container) | Adapter serving | ✗ (not started) | needs start | Use merge+serve instead |
| wp-bench repo | wp-bench eval | ✗ (not cloned) | — | Skip wp-bench (partial triage only) |
| scipy | eval_judge.py Spearman | ✓ | in env | — |
| openai client | All eval scripts | ✓ | in env | — |
| eval_toolbox container | eval scripts (EVAL-04) | Available via dgx_toolbox | eval-toolbox.sh | — |

**Missing dependencies with no fallback:**
- wp-bench repo must be cloned before wp-bench eval can run — this is a Wave 0 setup task

**Missing dependencies with fallback:**
- vLLM LoRA serving: if `--lora-modules` fails due to `modules_to_save`, fall back to merged model serving

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (project uses Python scripts; no test/ dir found) |
| Config file | none — no pytest.ini or conftest.py found |
| Quick run command | `python -m pytest tests/ -x -q` (if tests written) |
| Full suite command | `python -m eval.eval_gate --results-dir output/eval_triage/ratio_{r}` (functional gate) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | PHPCS pass rate >95% measured | Integration | `python -m eval.eval_gate --results-dir output/eval_triage/ratio_{r}` | ❌ Wave 0: output dir not yet created |
| EVAL-02 | Judge Spearman >0.85 measured | Integration | `python -m eval.eval_gate --results-dir output/eval_triage/ratio_{r}` | ❌ Wave 0: output dir not yet created |
| EVAL-03 | Security pass rate >98% measured | Integration | `python -m eval.eval_gate --results-dir output/eval_triage/ratio_{r}` | ❌ Wave 0: output dir not yet created |
| EVAL-04 | Scripts run via eval-toolbox container | Smoke | `dgx.execute("eval_toolbox", "python", "-m", "eval.eval_gate", "--help")` | ❌ Wave 0 |
| EVAL-05 | All 3 gates pass | Integration | `eval_gate.run_gate(results_dir=out_dir)` exits 0 | ❌ depends on eval completion |
| GATE-02 | Triage elimination logic applied correctly | Unit | `pytest tests/test_triage_logic.py` | ❌ Wave 0: needs test |

### Sampling Rate
- **Per task commit:** `python -m eval.eval_gate --help` (smoke — confirms scripts importable)
- **Per wave merge:** `python -m eval.eval_gate --results-dir output/eval_triage/ratio_{r}` (functional gate on results)
- **Phase gate:** Full eval complete + triage_decision.md written + GATE-02 applied

### Wave 0 Gaps
- [ ] `output/eval_triage/` — create directory structure before eval loop
- [ ] `wp-bench/` — clone `github.com/WordPress/wp-bench` and run setup
- [ ] `tests/test_triage_logic.py` — unit test for GATE-02 elimination logic (D-12)
- [ ] Verify vLLM LoRA loading works with adapters that have `modules_to_save` tensors

## Sources

### Primary (HIGH confidence)
- `/home/robert_li/Desktop/projects/wp-finetune/models/Qwen3-30B-A3B/config.json` — model architecture (48 layers, 128 experts, top-8, decoder_sparse_step=1)
- transformers 5.3.0 source: `Qwen3MoeSparseMoeBlock`, `Qwen3MoeTopKRouter`, `Qwen3MoeDecoderLayer` — inspected directly
- `eval/eval_gen.py`, `eval/eval_judge.py`, `eval/eval_gate.py` — complete scripts read in full
- `config/dgx_toolbox.yaml` — container config, ports, component paths
- `config/wp-bench.yaml` — wp-bench config confirming integration setup
- `adapters/tokenizer/` — verified `<wp_gen>`=151669, `<wp_judge>`=151670

### Secondary (MEDIUM confidence)
- `~/dgx-toolbox/inference/start-vllm.sh` — confirmed extra-args passthrough for `--enable-lora --lora-modules`
- MoE-Sieve paper (arxiv 2603.24044) — E_eff methodology, 10% Jaccard stability claim (from CONTEXT.md canonical refs)
- Memory reference `reference_wp_bench.md` — wp-bench execution model (Docker WordPress runtime, no LLM in loop)

### Tertiary (LOW confidence)
- vLLM LoRA + `modules_to_save` compatibility: inferred from vLLM docs pattern; not directly tested on this adapter format — flagged as Open Question 1

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified in env or container
- Architecture patterns: HIGH — router source code inspected directly, hook output verified
- Pitfalls: HIGH for code patterns (verified); MEDIUM for vLLM LoRA compatibility (not tested)
- E_eff formula: HIGH — matches published MoE-Sieve formula, numpy implementation standard

**Research date:** 2026-04-01
**Valid until:** 2026-05-01 (transformers API stable; vLLM LoRA API may change faster)
