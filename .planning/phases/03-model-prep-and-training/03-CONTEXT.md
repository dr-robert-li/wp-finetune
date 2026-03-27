# Phase 3: Model Prep and Training - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Download Qwen3-30B-A3B (native MoE), extend tokenizer with `<wp_gen>` and `<wp_judge>` task tokens, write evaluation suite, and fine-tune via Unsloth LoRA on DGX Spark. No MoE conversion needed — model is already a native MoE.

</domain>

<decisions>
## Implementation Decisions

### Base Model
- **Qwen3-30B-A3B** — native MoE, 128 experts, top-8 routing, ~3B active params
- Download from HuggingFace, no conversion step
- Fits DGX Spark 128GB unified memory (60GB BF16, 15GB QLoRA)
- Verified Unsloth support for fine-tuning

### Evaluation Suite — wp-bench + Custom
- **Primary eval: wp-bench** (`github.com/WordPress/wp-bench`) — the canonical WordPress AI benchmark
  - Execution tests: code generation graded by real WordPress runtime (static checks + runtime assertions)
  - Knowledge tests: multiple choice on WP APIs, security, hooks, REST, caching
  - Covers: hooks, REST API, database, caching, HTML API, block API, security, queries
  - NO Claude in the eval loop — solves circularity completely
  - wp-bench must be installed and runtime running before eval
- **Custom judge eval:** Compare model's judge scores against PHPCS/PHPStan results (no Claude)
  - Generate 500 held-out code samples, run model's `<wp_judge>` mode
  - Compare model's pass/fail verdict with PHPCS pass/fail as ground truth
  - Measure: Spearman correlation of model's overall_score vs PHPCS error count (inverted)
  - Security eval: test against known-vulnerable code samples from failed/ directory
- **Supplementary:** PHPCS pass rate on held-out test split (597 examples) for gen mode

### Training Config
- **Claude's discretion** — Claude picks LoRA rank, epochs, batch size, learning rate based on:
  - Dataset size: 5,958 examples (4,766 train)
  - Model size: 30B total, 3B active
  - DGX Spark: 128GB unified memory
  - Balance iteration speed vs thoroughness
- **Locked constraints:**
  - `modules_to_save=["embed_tokens", "lm_head"]` — special token embeddings must train
  - MoE load balancing loss monitored (no routing collapse)
  - W&B experiment tracking active
  - LoRA adapter kept separate until eval passes
  - bf16 training

### DGX Toolbox Integration (LOCKED)
- **All scripts MUST use `scripts/dgx_toolbox.py` resolver** — never hardcode ~/dgx-toolbox or any absolute paths
- **Toolbox location configurable:** `config/dgx_toolbox.yaml` → `dgx_toolbox_path` or `DGX_TOOLBOX_PATH` env var
- **Training:** Use `dgx.run("unsloth_studio")` to launch Unsloth container
- **Serving:** Use `dgx.run("vllm", model_path)` — endpoint via `dgx.vllm_endpoint()`
- **Eval:** Use `dgx.run("eval_toolbox")` — eval scripts run inside eval-toolbox container
- **LiteLLM proxy:** Use `dgx.run("litellm")` — wp-bench routes through `dgx.litellm_endpoint()`
- **Ports from config:** `dgx.port("vllm")` = 8020, `dgx.port("litellm")` = 4000, etc.
- **Transportability:** Projects are a paired set — clone both, point config, everything works
- **Import pattern:** `from scripts.dgx_toolbox import get_toolbox; dgx = get_toolbox()`

### Research Findings (LOCKED)
- **QLoRA is OFF-LIMITS for MoE** — Unsloth explicitly states BitsandBytes doesn't support MoE nn.Parameter in 4-bit. Must use `load_in_4bit=False` with BF16 LoRA.
- **Peak memory ~63GB** — fits DGX Spark 128GB with Unsloth FastLanguageModel (avoids page cache OOM)
- **output_router_logits=True** must be explicitly set for MoE load balancing loss to appear in W&B
- **modules_to_save merge is buggy** (Unsloth #3444) — keep adapter separate through Phase 3, merge in Phase 4 with verification roundtrip

### Claude's Discretion
- LoRA rank (r=32 vs r=64), alpha, dropout
- Number of epochs (1-3)
- Batch size and gradient accumulation
- Learning rate and scheduler
- BF16 LoRA only (QLoRA off-limits per research)
- Training script format (Jupyter notebook via Unsloth Studio or headless Python)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### wp-bench (evaluation)
- `https://github.com/WordPress/wp-bench` — Clone and install for eval
- Execution tests: `datasets/suites/wp-core-v1/execution/*.json` — 14 categories
- Knowledge tests: `datasets/suites/wp-core-v1/knowledge/*.json` — 21 categories
- Runtime: `runtime/` — WordPress sandbox for execution grading
- Config: `wp-bench.example.yaml` — model configuration template

### Training data
- `final_dataset/openai_train.jsonl` — 4,766 training examples in OpenAI format
- `final_dataset/openai_val.jsonl` — 595 validation examples
- `final_dataset/openai_test.jsonl` — 597 test examples (held-out for eval)
- `final_dataset/metadata.json` — dataset statistics

### Model
- `Qwen/Qwen3-30B-A3B` — HuggingFace model page
- Unsloth DGX Spark playbook: `github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/unsloth`

### DGX Toolbox Integration (MUST READ)
- `config/dgx_toolbox.yaml` — Configurable path to dgx-toolbox + all component paths, ports, shared dirs
- `scripts/dgx_toolbox.py` — Python resolver: `from scripts.dgx_toolbox import get_toolbox; dgx = get_toolbox()`
  - `dgx.run("unsloth_studio")` — launch training container
  - `dgx.run("vllm", model_path)` — start inference
  - `dgx.run("eval_toolbox")` — launch eval container
  - `dgx.vllm_endpoint()` — returns `http://localhost:{port}/v1`
  - `dgx.litellm_endpoint()` — returns `http://localhost:{port}/v1`
- **ALL scripts in this phase MUST import and use the resolver — NEVER hardcode paths or ports**

### Project config
- `config/judge_system.md` — Judge rubric (for custom judge eval design)
- `.planning/research/STACK.md` — Pinned library versions for DGX Spark container

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `final_dataset/openai_train.jsonl` — ready for Unsloth SFTTrainer ingestion
- `scripts/pipeline_orchestrator.py` — state tracking pattern reusable for training monitoring
- `config/judge_system.md` — rubric defines what the custom judge eval should measure

### Established Patterns
- All pipeline output in OpenAI messages format (`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`)
- Task tokens (`<wp_gen>`, `<wp_judge>`) already in the training data user messages
- DGX Toolbox components used via shell scripts in `~/dgx-toolbox/`

### Integration Points
- Training data → Unsloth SFTTrainer (reads OpenAI JSONL directly)
- Trained model → vLLM (:8020) and Ollama (:11434) for serving
- wp-bench → evaluation (runs against served model via API)
- W&B → experiment tracking (eval-toolbox container)

</code_context>

<specifics>
## Specific Ideas

- wp-bench (`github.com/WordPress/wp-bench`) is the canonical WordPress AI benchmark — user explicitly identified it as the accepted standard
- Eval circularity solved: WordPress runtime grades generated code, not Claude
- Custom judge eval uses PHPCS/PHPStan as ground truth, not Claude scores
- Training on DGX Spark via DGX Toolbox Unsloth Studio

</specifics>

<deferred>
## Deferred Ideas

- DPO/RLHF refinement — v2, after initial SFT results are evaluated
- Adversarial testing feedback loop — Phase D4, feed results back into training data for v2
- Multi-epoch hyperparameter search — v2, start with Claude's best-guess config for v1

</deferred>

---

*Phase: 03-model-prep-and-training*
*Context gathered: 2026-03-27*
