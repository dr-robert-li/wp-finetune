# Phase 3: Model Prep and Training - Research

**Researched:** 2026-03-27 (updated)
**Domain:** Qwen3-30B-A3B MoE fine-tuning, tokenizer extension, wp-bench evaluation
**Confidence:** HIGH (stack verified via Unsloth docs, DGX Spark community reports, HuggingFace model card)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Base model:** Qwen3-30B-A3B — native MoE, 128 experts, top-8 routing, ~3B active params. Download from HuggingFace, no conversion step.
- **QLoRA is OFF-LIMITS for MoE** — BitsandBytes does not support MoE nn.Parameter in 4-bit. Must use `load_in_4bit=False` with BF16 LoRA. This is not discretion — it is a hard constraint.
- **Primary eval:** wp-bench (`github.com/WordPress/wp-bench`) — execution tests graded by real WordPress runtime + knowledge multiple-choice. NO Claude in the eval loop.
- **Custom judge eval:** Compare model's `<wp_judge>` scores against PHPCS/PHPStan results. 500 held-out code samples, Spearman correlation of overall_score vs PHPCS error count (inverted). Security eval against known-vulnerable samples.
- **Supplementary eval:** PHPCS pass rate on held-out test split (597 examples) for gen mode.
- **Training locked constraints:**
  - `modules_to_save=["embed_tokens", "lm_head"]` — special token embeddings must train
  - MoE load balancing loss monitored (no routing collapse)
  - W&B experiment tracking active
  - LoRA adapter kept separate until eval passes
  - bf16 training
- **Task tokens:** `<wp_gen>` and `<wp_judge>` — already in training data user messages
- **DGX Toolbox resolver:** All scripts MUST use `from scripts.dgx_toolbox import get_toolbox; dgx = get_toolbox()`. Never hardcode `~/dgx-toolbox` or any absolute paths.
- **Merge strategy (bug FIXED):** unsloth-zoo PR #369 + PR #559 are in 2026.3.5. `merge_and_unload()` works. Still: save adapter separately first (defense-in-depth), then attempt merge with verification roundtrip (save → reload → test special tokens). vLLM `--lora-modules` as fallback if merge fails.
- **Data directory prefix:** All pipeline output is under `data/`. Training data at `data/final_dataset/openai_train.jsonl`.
- **Eval scripts location:** Keep in `eval/` directory, separate from `scripts/` pipeline scripts.

### Claude's Discretion

- LoRA rank (r=32 vs r=64), alpha, dropout
- Number of epochs (1-3)
- Batch size and gradient accumulation
- Learning rate and scheduler
- Training script format (Jupyter notebook via Unsloth Studio or headless Python)

### Deferred Ideas (OUT OF SCOPE)

- DPO/RLHF refinement — v2, after initial SFT results are evaluated
- Adversarial testing feedback loop — Phase D4, feed results back into training data for v2
- Multi-epoch hyperparameter search — v2, start with Claude's best-guess config for v1
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODL-01 | Qwen3-30B-A3B downloaded (native MoE, 128 experts, top-8 routing) | HuggingFace `snapshot_download`; 61.1 GB BF16 safetensors; requires 128 GB disk headroom |
| MODL-02 | Tokenizer extended with `<wp_gen>` and `<wp_judge>` special tokens | `add_special_tokens()` + mean initialization pattern; save tokenizer before LoRA setup |
| MODL-03 | Model embeddings resized and new token embeddings initialized (mean of existing) | `resize_token_embeddings(len(tokenizer))`; compute mean of existing embed rows; assign to new rows |
| MODL-04 | Smoke test passes — model loads, generates coherent text, task tokens are recognized | `tokenizer.encode("<wp_gen>")` must return a single token ID; generate 50-token sample |
| TRNG-01 | Unsloth LoRA SFT configured on DGX Spark (r=64, bf16, cosine LR) | `FastLanguageModel.get_peft_model` with `load_in_4bit=False`; `lora_alpha=128`; `lr_scheduler_type="cosine"` |
| TRNG-02 | LoRA config includes `modules_to_save=["embed_tokens", "lm_head"]` for special tokens | PEFT `LoraConfig.modules_to_save`; merge bug FIXED in unsloth-zoo 2026.3.5; still save adapter separately first |
| TRNG-03 | Training data loaded as 50/50 wp_gen/wp_judge multi-task mix | `data/final_dataset/openai_train.jsonl` (actual split: 40/60 gen/judge per metadata); shuffle; SFTTrainer reads OpenAI format natively |
| TRNG-04 | MoE load balancing loss monitored throughout training | `output_router_logits=True` in model kwargs; W&B logs `train/router_aux_loss` automatically |
| TRNG-05 | W&B experiment tracking active via eval-toolbox | `wandb.init()` in training script; `report_to="wandb"` in TrainingArguments |
| TRNG-06 | Training completes without OOM or divergence on DGX Spark | 63 GB peak for BF16 LoRA on Qwen3-30B-A3B; 128 GB UMA sufficient; use Unsloth gradient checkpointing |
| EVAL-01 | Custom eval script: PHPCS pass rate on 500 held-out generation tasks (target >95%) | Run vLLM-served model on `data/final_dataset/openai_test.jsonl` wp_gen examples; pipe output through `phpcs --standard=WordPress`; count errors=0 |
| EVAL-02 | Custom eval script: judge Spearman correlation on 500 held-out scored pairs (target >0.85) | `scipy.stats.spearmanr`; model overall_score vs inverted PHPCS error count as ground truth |
| EVAL-03 | Security pass rate measured on held-out tasks (target >98%) | Subset of EVAL-01: filter for security-critical patterns (nonce, SQL, escaping); check phpcs security-related sniffs |
| EVAL-04 | Eval scripts run via DGX Toolbox eval-toolbox container | `dgx.run("eval_toolbox")` via resolver; model served via `dgx.run("vllm", model_path)` on port `dgx.port("vllm")` = 8020 |
| EVAL-05 | All three quality gates pass before proceeding to deployment | Gate script checks EVAL-01 >= 95%, EVAL-02 >= 0.85, EVAL-03 >= 98%; returns non-zero exit if any fail |
</phase_requirements>

---

## Summary

Phase 3 covers four sequential stages: download and verify Qwen3-30B-A3B, extend the tokenizer with two special tokens and resize embeddings, fine-tune via Unsloth LoRA SFT on DGX Spark, then run the evaluation suite. The model is already a native MoE — no conversion step — which eliminates the biggest risk from the original design.

The critical hardware finding is that Qwen3-30B-A3B BF16 LoRA requires 63 GB VRAM (Unsloth's measured figure). DGX Spark's 128 GB unified memory accommodates this with headroom, but only because 4-bit QLoRA is off-limits for MoE models (BitsandBytes limitation confirmed in Unsloth docs). The community reference for DGX Spark MoE training (Qwen3.5-35B-A3B at rank 16) shows the pattern is proven, but memory management during model loading requires care — the model must shard directly to CUDA to avoid page cache doubling.

The evaluation story is the cleanest aspect of the phase: wp-bench uses WordPress itself as the grader with no LLM in the loop, and the custom judge eval uses PHPCS error counts as ground truth. Both avoid the circularity of the earlier Claude-scored eval approach. wp-bench connects via LiteLLM conventions to any OpenAI-compatible endpoint, which is exactly what vLLM :8020 provides. The merge bug that previously required keeping the adapter permanently separate is fixed in unsloth-zoo 2026.3.5 — the strategy is now: save adapter first (defense-in-depth), attempt merge with roundtrip verification, keep vLLM `--lora-modules` as fallback.

**Primary recommendation:** Use `FastLanguageModel` (not `AutoModelForCausalLM`) with `load_in_4bit=False`, r=32, `lora_alpha=64`, `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_up_proj","down_proj"]`, `modules_to_save=["embed_tokens","lm_head"]`, bf16, cosine LR 2e-4, 2 epochs. This fits comfortably in 63 GB and matches the community-proven DGX Spark pattern.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| unsloth | 2026.3.x (latest) | FastLanguageModel, LoRA, MoE kernels | Official DGX Spark integration; 1.8x faster than HF for MoE; only framework with MoE LoRA kernels for Qwen3 |
| unsloth-zoo | 2026.3.5+ | merge_and_unload fix | PR #369 + PR #559 both merged; modules_to_save corruption fixed |
| transformers | >=4.51.0 (pinned 4.56.2 in DGX playbook) | Model loading, tokenizer, Qwen3MoE architecture | `qwen3_moe` arch requires >=4.51.0; DGX Spark playbook pins 4.56.2 for Blackwell stability |
| trl | 0.26.1 (DGX playbook pin) | SFTTrainer with OpenAI format support | Required by Unsloth DGX playbook; SFTTrainer reads `{"messages": [...]}` JSONL directly |
| peft | >=0.14.0 | LoraConfig, modules_to_save | Required by Unsloth; handles embed_tokens/lm_head save during adapter creation |
| datasets | 4.3.0 (DGX playbook pin) | Dataset loading from JSONL | DGX playbook pin; `load_dataset("json", data_files=...)` reads openai_train.jsonl |
| accelerate | >=1.0.0 | Mixed precision, gradient accumulation | Required for bf16 TrainingArguments |
| wandb | latest | Experiment tracking, loss curves | `report_to="wandb"` in TrainingArguments; eval-toolbox has W&B pre-installed |
| scipy | >=1.11 | Spearman correlation for judge eval | `scipy.stats.spearmanr` for EVAL-02 |
| huggingface_hub | >=0.23.0 | Model download via snapshot_download | `snapshot_download("Qwen/Qwen3-30B-A3B")` with resume_download=True |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| bitsandbytes | 0.48.0 | Quantization (NOT used for MoE LoRA) | Only for inference quantization validation; NOT for training (QLoRA unsupported on MoE) |
| openai | >=1.30 | Client for eval scripts hitting vLLM :8020 | EVAL-01/02/03: call vLLM as OpenAI-compatible endpoint |
| subprocess / json / pathlib | stdlib | phpcs wrapper in eval scripts | No extra install; pipe phpcs CLI output to Python |

### wp-bench
| Component | Install | Purpose |
|-----------|---------|---------|
| wp-bench Python harness | `pip install -e ./python` inside repo | Orchestrates benchmark runs |
| wp-bench runtime | `cd runtime && npm install && npm start` | WordPress sandbox that grades generated code |
| Node.js + @wordpress/env | via npm | Docker-based WordPress environment for runtime |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Unsloth + SFTTrainer | Axolotl | Axolotl has richer config files but no official DGX Spark playbook; Unsloth is the proven path |
| bf16 LoRA | 4-bit QLoRA | QLoRA is explicitly unsupported for MoE in BitsandBytes; would corrupt gradients |
| r=32 | r=64 | r=64 uses more memory on MoE (~2x per expert layer); r=32 is adequate for SFT adaptation of 5,958 examples |
| scipy Spearman | pearsonr | Spearman is rank-based and handles non-normal score distributions; correct for ordinal scores |

**Installation (inside Unsloth Studio container):**
```bash
# Training stack (most pre-installed in DGX Toolbox Unsloth Studio container)
pip install unsloth unsloth_zoo
pip install "transformers==4.56.2" "trl==0.26.1" "datasets==4.3.0"
pip install "bitsandbytes==0.48.0" --no-deps
pip install peft accelerate wandb scipy

# wp-bench (in eval-toolbox container)
git clone https://github.com/WordPress/wp-bench
cd wp-bench
pip install -e ./python
cd runtime && npm install  # requires Node.js
```

---

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── download_model.py        # MODL-01: snapshot_download with resume
├── prepare_tokenizer.py     # MODL-02/03/04: add tokens, resize, mean init, smoke test
├── train_model.py           # Main training script (headless Python, TRNG-01 through TRNG-06)
└── dgx_toolbox.py           # Existing resolver — all scripts import from here

eval/                        # Evaluation scripts — keep SEPARATE from pipeline scripts
├── eval_gen.py              # EVAL-01/03: PHPCS pass rate + security rate
├── eval_judge.py            # EVAL-02: Spearman correlation against PHPCS
└── eval_gate.py             # EVAL-05: quality gate script, non-zero exit on fail

config/
├── train_config.yaml        # Externalised hyperparams (r, alpha, lr, epochs)
├── wp-bench.yaml            # wp-bench YAML pointing at vLLM :8020 via LiteLLM
└── dgx_toolbox.yaml         # Existing DGX Toolbox resolver config

data/                        # Pipeline output (gitignored)
├── final_dataset/
│   ├── openai_train.jsonl   # 4,766 training examples
│   ├── openai_val.jsonl     # 595 validation examples
│   ├── openai_test.jsonl    # 597 test examples (held-out for eval)
│   └── metadata.json        # Dataset statistics
└── checkpoints/             # Pipeline execution state

models/                      # Downloaded base model
└── Qwen3-30B-A3B/           # ~61.1 GB, 16 safetensors shards

adapters/                    # LoRA adapter checkpoint (kept separate until verified merge)
├── tokenizer/               # Extended tokenizer (saved before LoRA setup)
└── qwen3-wp/                # LoRA adapter weights + modules_to_save tensors

tests/                       # Existing test suite (pytest, 46 tests)
```

### Pattern 1: DGX Toolbox Resolver Usage
**What:** All scripts use the resolver to get component paths, ports, and endpoints
**When to use:** Any script that invokes DGX Toolbox components or needs ports/endpoints
**Example:**
```python
# Source: scripts/dgx_toolbox.py (already in project)
from scripts.dgx_toolbox import get_toolbox

dgx = get_toolbox()  # singleton; reads config/dgx_toolbox.yaml

# Launch training container
dgx.run("unsloth_studio")

# Start vLLM inference (model_path is positional arg to the shell script)
dgx.run("vllm", "./models/Qwen3-30B-A3B-merged")

# Get endpoint URLs (never hardcode ports)
vllm_url = dgx.vllm_endpoint()       # http://localhost:8020/v1
litellm_url = dgx.litellm_endpoint() # http://localhost:4000/v1

# Launch eval container
dgx.run("eval_toolbox")

# Check availability (skip DGX-dependent steps in CI)
if not dgx.available:
    print("DGX Toolbox not found — skipping container launch")
```

### Pattern 2: Tokenizer Extension with Mean Initialization
**What:** Add `<wp_gen>` and `<wp_judge>` as new special tokens with stable initialization
**When to use:** Before any LoRA setup — tokenizer must be saved first
**Example:**
```python
# Source: https://langcopilot.com/posts/2025-09-23-how-to-add-special-tokens-llms
# and PEFT modules_to_save docs
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-30B-A3B")
new_tokens = ["<wp_gen>", "<wp_judge>"]
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen3-30B-A3B", torch_dtype=torch.bfloat16, device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

# Mean initialization — prevents random-vector training instability
with torch.no_grad():
    mean_embedding = model.model.embed_tokens.weight[:-2].mean(dim=0)
    model.model.embed_tokens.weight[-2] = mean_embedding  # <wp_gen>
    model.model.embed_tokens.weight[-1] = mean_embedding  # <wp_judge>
    # lm_head shares weights with embed_tokens in Qwen3; no separate init needed
    # but modules_to_save=["embed_tokens", "lm_head"] still required to train both

tokenizer.save_pretrained("./adapters/tokenizer")
```

### Pattern 3: Unsloth MoE LoRA Setup
**What:** Configure FastLanguageModel for Qwen3-30B-A3B with correct MoE target modules
**When to use:** After tokenizer is saved; feed model to SFTTrainer
**Example:**
```python
# Source: https://unsloth.ai/docs/new/faster-moe
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/Qwen3-30B-A3B",
    max_seq_length=4096,
    load_in_4bit=False,      # REQUIRED for MoE — 4-bit not supported
    dtype=torch.bfloat16,
    model_kwargs={"output_router_logits": True},  # TRNG-04: MoE aux loss
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_up_proj", "down_proj"],   # MoE projections included
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",           # Memory-efficient activation storage
    modules_to_save=["embed_tokens", "lm_head"],   # LOCKED: train special token embeddings
    random_state=42,
)
```

### Pattern 4: SFTTrainer with OpenAI JSONL
**What:** Feed `data/final_dataset/openai_train.jsonl` directly to SFTTrainer
**When to use:** Training loop
**Example:**
```python
# Source: https://huggingface.co/docs/trl/sft_trainer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

train_dataset = load_dataset(
    "json",
    data_files="data/final_dataset/openai_train.jsonl",
    split="train"
)
val_dataset = load_dataset(
    "json",
    data_files="data/final_dataset/openai_val.jsonl",
    split="train"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        output_dir="./adapters/qwen3-wp",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # effective batch=8
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        report_to="wandb",
        max_seq_length=4096,
    ),
)
trainer.train()
```

### Pattern 5: Adapter Save + Merge with Verification Roundtrip
**What:** Save adapter first (defense-in-depth), then merge with special token verification
**When to use:** After training completes. Merge bug FIXED in unsloth-zoo 2026.3.5 — but still verify.
**Example:**
```python
# Step 1: Save adapter separately (defense-in-depth — always do this first)
model.save_pretrained("./adapters/qwen3-wp")
tokenizer.save_pretrained("./adapters/tokenizer")

# Step 2: Attempt merge
merged_path = "./models/Qwen3-30B-A3B-merged"
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)

# Step 3: Verification roundtrip — reload and test special tokens
from transformers import AutoTokenizer, AutoModelForCausalLM
verify_tok = AutoTokenizer.from_pretrained(merged_path)
wp_gen_ids = verify_tok.encode("<wp_gen>", add_special_tokens=False)
wp_judge_ids = verify_tok.encode("<wp_judge>", add_special_tokens=False)
assert len(wp_gen_ids) == 1, f"<wp_gen> must be single token after merge, got {wp_gen_ids}"
assert len(wp_judge_ids) == 1, f"<wp_judge> must be single token after merge, got {wp_judge_ids}"
print("Merge verification passed — merged model is valid")

# If verification fails: vLLM --lora-modules is the fallback (serve adapter directly)
```

### Pattern 6: wp-bench with Local vLLM Endpoint
**What:** Configure wp-bench YAML to target vLLM serving on :8020
**When to use:** After training and vLLM startup via `dgx.run("vllm", model_path)`
**Example:**
```yaml
# config/wp-bench.yaml
# Source: wp-bench.example.yaml + LiteLLM OpenAI-compatible docs
dataset:
  source: local
  name: wp-core-v1

models:
  - name: openai/qwen3-wp          # LiteLLM "openai/" prefix = custom endpoint
    api_base: "http://localhost:8020/v1"
    api_key: "none"

grader:
  kind: docker
  wp_env_dir: ./wp-bench/runtime

run:
  suite: wp-core-v1
  limit: null          # run all tests
  concurrency: 4

output:
  path: output/wp-bench-results.json
  jsonl_path: output/wp-bench-results.jsonl
```

### Pattern 7: Custom Judge Eval (Spearman Correlation)
**What:** Measure how well model's `<wp_judge>` scores correlate with PHPCS verdict
**When to use:** EVAL-02 / EVAL-03 — runs against `data/final_dataset/openai_test.jsonl`
**Example:**
```python
# Source: REQUIREMENTS.md + scipy docs
import json, subprocess, scipy.stats
from scripts.dgx_toolbox import get_toolbox
from openai import OpenAI

dgx = get_toolbox()
client = OpenAI(base_url=dgx.vllm_endpoint(), api_key="none")

results = []
for example in held_out_test_examples:  # 500 samples from data/final_dataset/openai_test.jsonl
    code = extract_code(example)
    # Run model in wp_judge mode via vLLM API
    model_score = call_model_judge(client, code)["overall_score"]
    # Run PHPCS as ground truth
    phpcs_errors = run_phpcs(code)  # count of error lines
    phpcs_score = max(0, 100 - phpcs_errors * 5)  # invert: errors -> score
    results.append((model_score, phpcs_score))

model_scores, phpcs_scores = zip(*results)
corr, pvalue = scipy.stats.spearmanr(model_scores, phpcs_scores)
# Target: corr >= 0.85
```

### Anti-Patterns to Avoid
- **Calling `load_in_4bit=True` for MoE LoRA:** BitsandBytes does not support MoE nn.Parameter in 4-bit. Must use `load_in_4bit=False`. This is not a recommendation — it is a hard failure mode.
- **Adding tokens AFTER LoRA wrapping:** Token extension must happen before `get_peft_model`. Reverse order breaks vocabulary alignment.
- **Hardcoding paths or ports:** Always use `from scripts.dgx_toolbox import get_toolbox`. Never write `~/dgx-toolbox`, `http://localhost:8020`, or port numbers directly in eval or training scripts.
- **Including `<think>` blocks in training targets:** Qwen3 thinking mode produces `<think>...</think>` prefixes. Training data must NOT include these. Set `enable_thinking=False` when applying chat template to any training examples.
- **Routing layer in LoRA target modules:** Unsloth disables router fine-tuning by default. Do not add `"gate"` (the routing gate, not the MoE MLP gate) to target_modules.
- **Skipping merge verification:** Even though the merge bug is fixed in 2026.3.5, always verify the roundtrip. The adapter save is the safety net; the merged model is the optimization.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OpenAI-format JSONL ingestion | Custom data loader | `SFTTrainer` with `load_dataset("json", ...)` | SFTTrainer natively unpacks `{"messages": [...]}` format |
| LoRA adapter + modules_to_save fusion | Custom model save logic | `model.save_pretrained(adapter_dir)` via PEFT | PEFT handles split saving of LoRA weights + full embed_tokens/lm_head correctly |
| Spearman correlation | Custom rank comparison | `scipy.stats.spearmanr` | Well-tested, handles ties, returns p-value |
| WordPress code execution for eval | Custom PHP sandbox | wp-bench runtime | WordPress itself grades code; handles static analysis + runtime assertions |
| W&B integration | Custom logging loop | `report_to="wandb"` in SFTConfig | Zero-code integration; logs train/eval loss, learning rate, router_aux_loss automatically |
| Resume interrupted training | Custom checkpoint logic | `SFTConfig(resume_from_checkpoint=True)` + HF checkpoint format | Built-in resumption from `trainer.save_checkpoint()` |
| DGX Toolbox path resolution | Hardcoded paths | `scripts/dgx_toolbox.py` resolver | Already in project; handles env var override, config file, default fallback |

**Key insight:** The training stack (Unsloth+TRL+SFTTrainer) handles every heavy-lift data and training concern. Phase 3 custom code is thin glue: tokenizer extension, eval metrics computation, and orchestration scripts that use the resolver.

---

## Common Pitfalls

### Pitfall 1: Page Cache OOM During Model Loading on DGX Spark
**What goes wrong:** Loading 61 GB safetensors via standard `from_pretrained` causes mmap page cache (~67 GB) plus CUDA tensors (~67 GB) to exceed 128 GB UMA, causing OOM at ~66% utilization.
**Why it happens:** Linux mmap keeps loaded shards in page cache even after tensors move to CUDA. Both copies coexist briefly.
**How to avoid:** Use `FastLanguageModel.from_pretrained` (Unsloth handles eager shard release internally). If using standard HF loading, patch with `posix_fadvise(POSIX_FADV_DONTNEED)` after each shard. Community reference: DGX Spark forum thread on Qwen3.5-35B-A3B.
**Warning signs:** OOM error with >60% memory consumed reported, no Python OOM exception (it's CUDA/kernel level).

### Pitfall 2: Merge Verification Failure (Bug Fixed, Verification Still Required)
**What goes wrong:** After training with `modules_to_save=["embed_tokens", "lm_head"]`, merged model fails the special token roundtrip test.
**Why it happens:** The merge bug is FIXED in unsloth-zoo 2026.3.5 (PR #369 + PR #559). However, version drift or container cache issues could expose an older version. The roundtrip test catches this before the merged model is used.
**How to avoid:** Always save adapter separately first. Always run the verification roundtrip after merge. If roundtrip fails, serve the adapter with vLLM `--lora-modules` — do not serve the corrupted merged model.
**Warning signs:** After merge, `model.generate()` produces incoherent output; task tokens decode as unknown or multi-token sequences; `tokenizer.encode("<wp_gen>")` returns more than one token ID.

### Pitfall 3: Thinking Mode Contamination
**What goes wrong:** Qwen3 has built-in thinking mode enabled by default. If `enable_thinking` is not set to `False` during chat template application, the model produces `<think>...</think>` prefixes that appear in training targets and confuse SFT.
**Why it happens:** Qwen3's default chat template enables thinking. The training JSONL was generated without thinking blocks, so contamination happens only if chat template is applied incorrectly at training time.
**How to avoid:** Apply `tokenizer.apply_chat_template(messages, enable_thinking=False)` in any preprocessing step. The raw `data/final_dataset/openai_train.jsonl` already lacks think blocks — don't re-apply the chat template unless necessary; SFTTrainer's `dataset_text_field` approach skips re-application.
**Warning signs:** Training loss spikes early; generated outputs start with `<think>` despite SFT training.

### Pitfall 4: wp-bench Runtime Not Running During Eval
**What goes wrong:** `wp-bench run` hangs or reports all execution tests as errors.
**Why it happens:** The WordPress grader runtime (`runtime/`) must be started separately (`npm start`) and must have Docker available for `@wordpress/env`.
**How to avoid:** Start runtime before any wp-bench eval; verify with the runtime-specific health check. Run runtime startup as a pre-step in the eval workflow.
**Warning signs:** All execution test scores are 0 or error; knowledge tests still pass (they don't need the runtime).

### Pitfall 5: Dataset Split Mismatch (40/60 not 50/50)
**What goes wrong:** REQUIREMENTS.md TRNG-03 says "50/50 wp_gen/wp_judge multi-task mix" but `data/final_dataset/metadata.json` shows actual ratio is 40/60 (2383 gen / 3575 judge).
**Why it happens:** Export script enforces 40/60 target ratio per the skill configuration (40% gen, 60% judge).
**How to avoid:** Use the dataset as-is. The 40/60 ratio is intentional and correct — more judge examples improve rubric calibration. Document actual ratio in W&B run config. Do not re-balance.
**Warning signs:** If you explicitly try to re-balance to 50/50, you will discard valid judge examples.

### Pitfall 6: Router Auxiliary Loss Not Appearing in W&B
**What goes wrong:** W&B logs only show `train/loss` and `eval/loss`; `router_aux_loss` is absent.
**Why it happens:** `output_router_logits` defaults to `False` in Qwen3MoE config. Without it, auxiliary loss is not computed or surfaced.
**How to avoid:** Pass `model_kwargs={"output_router_logits": True}` to `FastLanguageModel.from_pretrained` or set it on `model.config` after loading. Confirm presence in first 10 log steps.
**Warning signs:** Training loss decreases normally but routing behavior is unmonitored; possible silent expert collapse.

### Pitfall 7: Wrong Data Paths (Bare vs data/ Prefix)
**What goes wrong:** Script fails with FileNotFoundError because it looks for `final_dataset/openai_train.jsonl` instead of `data/final_dataset/openai_train.jsonl`.
**Why it happens:** Phase 3 plans may reference the old path structure from before the `data/` prefix standardization.
**How to avoid:** All pipeline output is under `data/`. The canonical paths are:
  - `data/final_dataset/openai_train.jsonl`
  - `data/final_dataset/openai_val.jsonl`
  - `data/final_dataset/openai_test.jsonl`
  - `data/final_dataset/metadata.json`
**Warning signs:** Any script that opens `final_dataset/` or `phase1_extraction/` without the `data/` prefix will fail.

---

## Code Examples

Verified patterns from official sources:

### Model Download with Resume
```python
# Source: huggingface_hub docs
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-30B-A3B",
    local_dir="./models/Qwen3-30B-A3B",
    resume_download=True,   # Safe to re-run; continues interrupted download
    ignore_patterns=["*.msgpack", "*.h5"],  # Skip non-safetensors formats
)
# Expected: ~61.1 GB across 16 safetensors shards
```

### Smoke Test (MODL-04)
```python
# Verify: model loads, generates text, task tokens recognized
token_ids = tokenizer.encode("<wp_gen>", add_special_tokens=False)
assert len(token_ids) == 1, f"<wp_gen> must be a single token, got {token_ids}"
token_ids = tokenizer.encode("<wp_judge>", add_special_tokens=False)
assert len(token_ids) == 1, f"<wp_judge> must be a single token, got {token_ids}"

# Quick generation test
prompt = "<wp_gen> Write a WordPress function that returns the current user's email."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50)
decoded = tokenizer.decode(out[0], skip_special_tokens=False)
assert len(decoded) > 10, "Generation appears broken"
print("Smoke test passed:", decoded[:100])
```

### Quality Gate Script (EVAL-05)
```python
# Gate: returns exit code 1 if any threshold fails
# Located at: eval/eval_gate.py
import sys

PHPCS_PASS_TARGET = 0.95    # EVAL-01
SPEARMAN_TARGET   = 0.85    # EVAL-02
SECURITY_TARGET   = 0.98    # EVAL-03

results = load_eval_results()
failed = []
if results["phpcs_pass_rate"] < PHPCS_PASS_TARGET:
    failed.append(f"PHPCS pass rate {results['phpcs_pass_rate']:.2%} < {PHPCS_PASS_TARGET:.0%}")
if results["spearman_corr"] < SPEARMAN_TARGET:
    failed.append(f"Spearman corr {results['spearman_corr']:.3f} < {SPEARMAN_TARGET}")
if results["security_pass_rate"] < SECURITY_TARGET:
    failed.append(f"Security pass rate {results['security_pass_rate']:.2%} < {SECURITY_TARGET}")

if failed:
    print("GATE FAILED:", "\n".join(failed))
    sys.exit(1)
print("All quality gates passed. Proceeding to Phase 4.")
sys.exit(0)
```

### DGX Toolbox Resolver in Eval Script
```python
# Source: scripts/dgx_toolbox.py (existing in project)
# Pattern used in eval/eval_gen.py and eval/eval_judge.py
from scripts.dgx_toolbox import get_toolbox
from openai import OpenAI

dgx = get_toolbox()
# vllm_endpoint() reads port from config/dgx_toolbox.yaml → ports.vllm = 8020
client = OpenAI(base_url=dgx.vllm_endpoint(), api_key="none")

# If DGX Toolbox is unavailable (CI environment), fail clearly
if not dgx.available:
    raise EnvironmentError(
        "DGX Toolbox not found. Eval scripts require a running vLLM instance. "
        f"Set DGX_TOOLBOX_PATH or update config/dgx_toolbox.yaml. "
        f"Current path: {dgx.path}"
    )
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Dense Qwen3-8B + CMoE conversion | Qwen3-30B-A3B (native MoE) | Phase 3 context decision 2026-03-27 | Eliminates 5-min conversion step; removes CMoE research-code risk; 10x more parameters |
| Claude-scored eval (circularity) | wp-bench WordPress runtime eval | Phase 3 context decision 2026-03-27 | No LLM in eval loop; grader is deterministic |
| `AutoModelForCausalLM.from_pretrained` | `FastLanguageModel.from_pretrained` | Unsloth 2025.x | 1.8x faster training; proper MoE kernel; handles page cache memory |
| LoRA on attention only | LoRA on attention + MoE MLP projections | Unsloth faster-moe update 2025 | Covers expert weights; enables true MoE adaptation |
| Keep adapter permanently separate (merge bug) | Save adapter + attempt merge + verify roundtrip | unsloth-zoo 2026.3.5 (PR #369 + PR #559) | Merge now works; Phase 4 can use merged model directly |
| `final_dataset/` bare path | `data/final_dataset/` prefixed path | Data directory standardization 2026-03-27 | All pipeline output under `data/`; scripts must use this prefix |

**Deprecated/outdated:**
- CMoE (arxiv:2502.04416): was planned for Qwen3-8B dense-to-MoE. No longer relevant — using native MoE.
- QLoRA (4-bit) for MoE training: BitsandBytes limitation; use BF16 full-precision LoRA instead.
- STACK.md references to CMoE/ToMoE: those sections are obsolete for Phase 3.
- Bare `final_dataset/` paths without `data/` prefix: update all scripts to use `data/final_dataset/`.

---

## Open Questions

1. **wp-bench `api_base` YAML key exact syntax**
   - What we know: wp-bench uses LiteLLM conventions; LiteLLM supports `openai/model-name` prefix + `api_base`
   - What's unclear: Whether the `api_base` key is a direct model-level config in wp-bench.yaml or requires a `litellm_params` nesting
   - Recommendation: Clone wp-bench before training, inspect `python/wp_bench/config.py` and `wp-bench.example.yaml` directly to confirm exact YAML key names. Fallback: use LiteLLM proxy in front of vLLM and point wp-bench at the proxy via `dgx.litellm_endpoint()`.

2. **output_router_logits interaction with Unsloth's FastLanguageModel**
   - What we know: Setting `output_router_logits=True` in model config makes TRL log auxiliary loss
   - What's unclear: Whether Unsloth's `FastLanguageModel` wrapper passes model_kwargs through to the underlying config, or requires setting `model.config.output_router_logits = True` post-load
   - Recommendation: Set it both ways to be safe: pass in `model_kwargs` to `from_pretrained` AND set `model.config.output_router_logits = True` after loading.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing; ~46 tests passing as of 2026-03-27) |
| Config file | none (standard pytest discovery) |
| Quick run command | `python3 -m pytest tests/ -x -q` |
| Full suite command | `python3 -m pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODL-01 | Model directory exists with expected shard count | smoke | `pytest tests/test_train_model.py::test_model_downloaded -x` | ❌ Wave 0 |
| MODL-02 | Tokenizer has `<wp_gen>` and `<wp_judge>` tokens | unit | `pytest tests/test_prepare_tokenizer.py::test_special_tokens_added -x` | ❌ Wave 0 |
| MODL-03 | New token embeddings initialized to mean (not random) | unit | `pytest tests/test_prepare_tokenizer.py::test_embeddings_mean_init -x` | ❌ Wave 0 |
| MODL-04 | Smoke test: task tokens are single-token IDs | unit | `pytest tests/test_prepare_tokenizer.py::test_smoke_single_token_ids -x` | ❌ Wave 0 |
| TRNG-01 | LoRA config r=32 (or r=64), bf16, cosine LR asserted in config | unit | `pytest tests/test_train_model.py::test_lora_config_params -x` | ❌ Wave 0 |
| TRNG-02 | modules_to_save contains embed_tokens and lm_head | unit | `pytest tests/test_train_model.py::test_modules_to_save -x` | ❌ Wave 0 |
| TRNG-03 | Training dataset loads from data/final_dataset/ with correct schema | unit | `pytest tests/test_train_model.py::test_dataset_schema -x` | ❌ Wave 0 |
| TRNG-04 | output_router_logits=True in model config | unit | `pytest tests/test_train_model.py::test_router_logits_enabled -x` | ❌ Wave 0 |
| EVAL-01 | PHPCS eval script runs on 3 sample outputs without crashing | integration | `pytest tests/test_eval_gen.py::test_phpcs_eval_runs -x` | ❌ Wave 0 |
| EVAL-02 | Spearman correlation computation correct on known values | unit | `pytest tests/test_eval_judge.py::test_spearman_computation -x` | ❌ Wave 0 |
| EVAL-03 | Security eval correctly identifies vulnerable code samples | unit | `pytest tests/test_eval_gen.py::test_security_rate_detection -x` | ❌ Wave 0 |
| EVAL-05 | Gate script exits 1 on failed thresholds, 0 on pass | unit | `pytest tests/test_eval_gate.py::test_gate_pass -x tests/test_eval_gate.py::test_gate_fail -x` | ❌ Wave 0 |

Note: TRNG-05 (W&B active), TRNG-06 (no OOM on DGX Spark), and EVAL-04 (DGX Toolbox container execution) are manual/environment checks only — not unit testable without live hardware.

### Sampling Rate
- **Per task commit:** `python3 -m pytest tests/ -x -q`
- **Per wave merge:** `python3 -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_prepare_tokenizer.py` — covers MODL-02, MODL-03, MODL-04
- [ ] `tests/test_train_model.py` — covers MODL-01, TRNG-01, TRNG-02, TRNG-03, TRNG-04
- [ ] `tests/test_eval_gen.py` — covers EVAL-01, EVAL-03 (uses mock phpcs subprocess)
- [ ] `tests/test_eval_judge.py` — covers EVAL-02 (pure Python, no model needed)
- [ ] `tests/test_eval_gate.py` — covers EVAL-05 (checks exit codes)

---

## Sources

### Primary (HIGH confidence)
- [Unsloth Faster MoE Docs](https://unsloth.ai/docs/new/faster-moe) — MoE LoRA configuration, `load_in_4bit=False` requirement, 63 GB memory figure, router layer exclusion, target modules
- [Unsloth Qwen3 Guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) — `enable_thinking=False` chat template, `FastLanguageModel`, Qwen3-30B-A3B VRAM figure
- [HuggingFace Qwen3-30B-A3B model card](https://huggingface.co/Qwen/Qwen3-30B-A3B) — Architecture: 48 layers, 128 experts, top-8 routing, 32k native context, requires transformers>=4.51.0; 61.1 GB BF16
- [TRL SFTTrainer Docs](https://huggingface.co/docs/trl/sft_trainer) — LoraConfig parameters, modules_to_save behavior
- [wp-bench GitHub](https://github.com/WordPress/wp-bench) — Repository structure, installation steps, YAML config structure, LiteLLM model naming conventions
- `scripts/dgx_toolbox.py` — Project-local resolver; `get_toolbox()`, `vllm_endpoint()`, `litellm_endpoint()`, `port()`, `run()` API confirmed by reading source
- `config/dgx_toolbox.yaml` — Ports: vllm=8020, litellm=4000, open_webui=12000, ollama=11434; components confirmed

### Secondary (MEDIUM confidence)
- [DGX Spark Qwen3.5-35B-A3B Forum Post](https://forums.developer.nvidia.com/t/bf16-lora-fine-tuning-of-qwen3-5-35b-a3b-on-dgx-spark-no-quantization-required/363268) — Page cache OOM pattern, `posix_fadvise` workaround, peak memory ~72 GB, batch_size=1
- [kreuzhofer/dgx-spark-unsloth-qwen3.5-training](https://github.com/kreuzhofer/dgx-spark-unsloth-qwen3.5-training) — Real DGX Spark training config: r=16, lr=2e-4, batch=2, grad_accum=4, target_modules=all attn+MLP
- [Unsloth Issue #3444 — modules_to_save merge corruption](https://github.com/unslothai/unsloth/issues/3444) — Confirmed bug; fix in unsloth-zoo PR #369 (merged 2026-01-30) + PR #559 (merged 2026-03-24)
- [How to Add Special Tokens Safely](https://langcopilot.com/posts/2025-09-23-how-to-add-special-tokens-llms) — Mean initialization rationale and implementation pattern
- [Qwen3 Technical Report (arxiv:2505.09388)](https://arxiv.org/pdf/2505.09388) — Global-batch load balancing loss in Qwen3 pre-training; no shared experts (unlike Qwen2.5-MoE)

### Tertiary (LOW confidence — verify before relying on)
- wp-bench YAML `api_base` exact key syntax for local LiteLLM endpoint — inferred from LiteLLM docs, not directly confirmed in wp-bench source

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — Unsloth docs + HuggingFace model card + DGX Spark community verified
- Architecture patterns: HIGH — Code examples from official Unsloth docs + PEFT docs + project source confirmed
- Pitfalls: HIGH — Most from confirmed bug reports and community DGX Spark posts; merge fix status confirmed via CONTEXT.md
- wp-bench local endpoint config: MEDIUM — Structure verified from repo README; exact YAML key for `api_base` needs source inspection

**Research date:** 2026-03-27
**Valid until:** 2026-04-27 (Unsloth releases frequently; re-check MoE docs if > 30 days)
