# Engineering Journal — wp-qwen3-moe

Decisions, reasoning, and observations logged as the project evolves.

---

## 2026-03-28 — Reflection: atomic composable skills as an architectural pattern

### What I'm actually building

I came into this project to fine-tune a WordPress model. Somewhere along the way, I started building something else without realising it: an architectural pattern for autonomous agent execution.

The three-layer stack that emerged from the training pipeline refactoring:

```
Skill (intent + recovery logic)
  → dgx_toolbox.py (resolves paths, validates state, handles errors)
    → Shell/Docker commands (generated dynamically, not hardcoded)
```

This isn't just a training pipeline. It's a general pattern: define *what* needs to happen and *how to recover from failure*, give agents a set of tools and parameters, and let them fill in the actual execution. The skills in `docs/` demonstrate this — they range from data pipeline orchestration (`run-data-pipeline.md`) to model training (`run-training.md`) to system monitoring (`observe-training.md`) to self-introspection (`review-telemetry.md`). Same pattern, completely different domains.

### Outcomes-driven, not functionally-driven

Traditional automation is functionally-driven: write a script that does step 1, then step 2, then step 3. If step 2 fails, the script fails. If the environment changes, the script breaks.

What I've ended up with is outcomes-driven: define the desired outcome ("model is trained"), define the preconditions ("model downloaded, tokenizer extended, memory sufficient"), define recovery paths ("if container dies, restart with mounts; if deps missing, install"), and let the agent figure out the sequencing. The skill doesn't *do* the work — it describes what done looks like and how to get unstuck.

The idempotency guards are the key enabler. Every step checks "is this already done?" before executing. This means an agent can be interrupted, restarted, or even replaced mid-execution, and it just picks up from where things left off. The skill doesn't need to know *which* agent ran previously or *how far* it got — the filesystem state is the checkpoint.

### It's like IKEA, but for software

IKEA gives you a bill of materials, assembly instructions with diagrams, and a bag of screws. You supply the labour and the Allen key. The instructions don't *do* the assembly — they describe the outcome of each step and the order of operations. The person building it fills in the motor skills.

These skills are the same thing. The skill file is the assembly instructions. The YAML config is the bill of materials. The agents are the person with the Allen key. They don't get frustrated, they don't misread the instructions (though they may misinterpret them — an important distinction), and they don't call customer support to complain.

If IKEA-built furniture is still furniture, isn't agent-built software still software?

### Why this matters beyond this project

The pattern is generally applicable:

- **Pipeline execution**: `run-data-pipeline.md` spawns agents to judge code, generate synthetics, score examples — each agent batch is independent, idempotent, and resumable
- **Infrastructure management**: `run-training.md` orchestrates container lifecycle, dependency installation, model download, training, and merge — each step validates before proceeding
- **Monitoring and alerting**: `observe-training.md` spawns a team of 6 specialised observers that independently track GPU metrics, thermals, training progress, disk I/O, checkpoint integrity, and container health
- **Self-introspection**: `review-telemetry.md` reads the output of the monitoring agents and produces a consolidated summary

The same Skill → Engine → Dynamic Commands stack could be applied to deploying a web application, running a CI/CD pipeline, managing a fleet of microservices, or building and testing software. The skill defines intent; the engine validates and resolves; the agents execute. Swap out the YAML config and you have a different project. Swap out the engine and you have a different infrastructure. The skills stay the same.

I didn't set out to build this. It emerged from repeatedly hitting the same problem — brittle shell scripts breaking silently — and iterating toward something that fails loudly, recovers gracefully, and doesn't need me watching it.

---

## 2026-03-28 — Refactor: dgx_toolbox.py as project-agnostic execution engine

### The problem

The training pipeline had accumulated a lot of manual handling: bash scripts with hardcoded container names, `docker exec` commands with hardcoded mount paths, pip install commands with hardcoded package lists. While repeatable, this was incredibly brittle — if a path changed, a container name changed, or DGX Toolbox restructured, the shell script would break silently. No error, just wrong behavior or a cryptic failure deep in a 6-12 hour training run.

I paused the training pipeline mid-download to fix this before continuing. The idempotency guards mean I can resume from exactly where I left off.

### Before: 8 hardcoded couplings

The original `dgx_toolbox.py` had project-agnostic infrastructure (path resolution, component lookup) mixed with wp-finetune-specific assumptions:

1. `CONTAINER_MAP` — hardcoded to 3 containers with wp-finetune workdirs
2. `_check_training_data` — hardcoded path `data/final_dataset/openai_train.jsonl`
3. `_check_config` — hardcoded path `config/train_config.yaml`
4. `_check_mounted` — checked for `train_config.yaml` specifically
5. `_check_deps` — hardcoded `import unsloth,trl,peft,...`
6. `status_report` artifacts — hardcoded model/adapter/dataset paths
7. `_install_deps` extras — hardcoded `wandb, peft, hf_transfer`
8. `PROJECT_ROOT` — assumed it lived inside the consuming project

### After: config-driven engine

All 8 couplings moved to `config/dgx_toolbox.yaml`. The Python module is now a pure execution engine that reads everything from config:

```yaml
# Container definitions (was CONTAINER_MAP global)
containers:
  unsloth_studio:
    container_name: unsloth-studio
    component: unsloth_studio
    workdir: /workspace/wp-finetune

# Validation paths (was hardcoded in _check_* methods)
validation_paths:
  training_data: data/final_dataset/openai_train.jsonl
  config: config/train_config.yaml

# Required imports (was hardcoded in _check_deps)
required_imports: [unsloth, trl, peft, datasets, wandb, yaml, scipy]

# Status artifacts (was hardcoded in status_report)
status_artifacts:
  model_downloaded: { path: models/Qwen3-30B-A3B/config.json, type: file }
  model_shards: { path: models/Qwen3-30B-A3B, type: glob, pattern: "*.safetensors" }
  ...
```

The Python engine (639 lines) now provides:
- `validate()` — named checks (`"toolbox"`, `"memory:70"`, `"container:unsloth_studio"`, `"deps:unsloth_studio"`) with structured `CheckResult` objects
- `ensure_ready()` — full lifecycle: start container → wait → check mount → install deps → validate
- `execute()` — container exec with idempotency checks, timing, and structured `ExecResult`
- `status_report()` — config-driven artifact checks for telemetry agents to consume

### The three-layer architecture

```
Skill (intent + recovery logic)          ← 8 skills in .claude/skills/
  → dgx_toolbox.py (resolve + validate)  ← 639-line config-driven engine
    → Shell/Docker commands (dynamic)      ← config/dgx_toolbox.yaml
```

Skills define *what* to do and *when* to retry. The Python engine handles *how* — resolving paths, validating preconditions, generating Docker commands dynamically. The YAML config holds all project-specific values. Changing a container name, adding a dependency, or pointing to a different training dataset is a YAML edit, not a code change.

### Making it reusable

Once the couplings moved to YAML, the engine became project-agnostic. Any project can supply its own `dgx_toolbox.yaml` and use the same Python engine. Example files were created in `examples/` showing how an external project would configure it. The core API is the same:

```python
from scripts.dgx_toolbox import get_toolbox
dgx = get_toolbox()
dgx.ensure_ready("unsloth_studio")
dgx.execute("unsloth_studio", "python", "-m", "scripts.train_model")
```

### Lessons learned

1. **Config over code for anything project-specific.** If a value might differ between projects (container names, paths, package lists), it belongs in YAML, not Python. The engine should be dumb about the domain and smart about execution.

2. **Pause and fix the foundation mid-pipeline.** Idempotency guards made it safe to stop, refactor, and resume. Without idempotency, I'd have been trapped — unable to fix the architecture without losing progress. This validates the earlier decision to make every step skip-safe.

3. **Structured validation beats silent failure.** The old shell script would fail deep in execution with a cryptic Docker error. The new engine runs all precondition checks upfront and returns structured `CheckResult` objects with actionable messages. Failing fast with a clear "training data not found at X" beats failing 2 hours in with "No such file."

---

## 2026-03-28 — Agentic telemetry framework: observability across containers and pipeline stages

### The problem

Training runs inside the Unsloth Studio Docker container on DGX Spark. The host sees GPU metrics via `nvidia-smi`, but training progress (loss curves, gradient norms, checkpoint saves) is only visible inside the container via `docker logs` and `docker exec`. System-level signals (disk I/O, thermal throttling, memory pressure) live on the host. There's no single place to look — I'd have to manually run `nvidia-smi`, `docker logs`, `docker stats`, `iostat`, and check adapter files in separate terminals, then mentally correlate the signals.

During the first training attempt, I manually spawned 7 background Claude Code agents to cover different monitoring concerns. It worked — each agent polled its signals, and I could check on them periodically. But it was ad-hoc: agents had to be re-spawned on every session, their prompts were written from scratch each time, and their output was scattered across temporary files. If I wanted to review what happened during a 6-hour training run, there was nothing persistent to look at.

### The solution: stage-specific telemetry skills

I encoded the monitoring patterns as reusable Claude Code skills — one per pipeline stage, each spawning a specialized team of background observer agents. The agents write append-only markdown reports to `telemetry/{stage}/{timestamp}/`, giving me both real-time visibility (tail the file) and a post-hoc audit trail.

**Skills created:**

| Skill | Agents | Stage |
|-------|--------|-------|
| `/observe-data-pipeline` | 3 (progress, system-resources, disk-io) | Data pipeline |
| `/observe-training` | 6 (gpu, thermal, training-metrics, disk-io, checkpoint, container) | Training |
| `/observe-evaluation` | 3 (eval-progress, gpu-metrics, result-tracking) | Evaluation |
| `/observe-packaging` | 3 (quantization-progress, file-integrity, size-tracking) | Packaging |
| `/observe-inference` | 5 (latency, throughput, gpu-util, memory, error-rates) | Serving |
| `/review-telemetry` | 0 (reads all reports, produces summary) | Any |

Each agent has concrete WARNING/CRITICAL thresholds (e.g., GPU temp > 80C, loss increasing for 3+ readings, disk > 85%) and a stop mechanism (`_stop` file). The execution skills (`run-training`, `run-data-pipeline`) now reference the relevant telemetry skill as an optional Step 0.

### Why agent teams vary by stage

Not every stage needs the same monitoring. The agent team composition is driven by a checklist:

- Uses GPU? → add gpu-metrics, possibly thermal-throttling
- Runs > 30 min? → add system-resources
- Writes large files? → add disk-io, file-integrity
- Runs in Docker? → add container-monitor
- Has checkpoints? → add checkpoint-integrity
- Has progress metric? → add stage-specific progress observer
- Serves network? → add latency, throughput, error-rates

Training needs all 6 concerns (GPU-heavy, Docker, long-running, checkpoints). The data pipeline only needs 3 (CPU-bound, no GPU, no Docker). Inference needs 5 (network-facing, latency-sensitive). This keeps the agent count proportional to the actual failure modes.

### Why this matters

The model trains for 6-12 hours unsupervised. Without structured telemetry, I'd either have to babysit it or discover problems after the fact with no diagnostic data. The framework means I say `/observe-training`, walk away, and come back to a full report — or say `/review-telemetry` mid-run for a consolidated status. Each new skill I create just needs to assess which agent team it needs using the checklist.

---

## 2026-03-28 — First training run failure: torch_dtype deprecation and pipeline hardening

### What happened

The first training run failed because `torch_dtype` has been fully deprecated in the current PyTorch/Unsloth stack. It's been a while since I've trained a model and I missed this — a simple mistake, but one that cost a failed run on the DGX Spark.

### Fix and hardening

After fixing the `torch_dtype` issue, I took the opportunity to harden the entire training pipeline:

1. **Stitched all Phase 3 scripts into a single Claude Code skill** (`docs/run-training.md`) — download, tokenizer prep, training, and merge now run as one atomic flow
2. **Made each step idempotent** — every script checks whether its output already exists and skips if so:

| Step | Skip condition | Re-run behavior |
|------|---------------|-----------------|
| Download model | Safetensors shards exist | Skips entirely |
| Extend tokenizer | `adapters/tokenizer/` has special tokens | Skips entirely |
| Train model | `adapter_config.json` exists | Skips (use `--resume` for partial) |
| Merge adapter | Merged model passes token verification | Skips entirely |

3. **Checkpoint-based resumability** — if training crashes mid-epoch, the next run picks up from the last checkpoint rather than starting over
4. **Memory pre-check** added to `train_model.py` — reads `/proc/meminfo` for available RAM (70GB minimum required). If insufficient: shows top memory consumers (processes + Docker containers), prints actionable suggestions (stop containers, prune Docker), and blocks training (`exit 1`) until memory is freed

The idempotency pattern is the same one that worked well in the data pipeline: check output → skip if exists → run if missing → verify output → proceed. This means a single "run training" command always does the right thing regardless of where the pipeline last stopped.

### Near-miss: memory-hungry containers

I almost started training without clearing memory-hungry containers from previous work. The DGX Spark had 8 containers running — vLLM, Open-WebUI, LiteLLM, n8n, and several unnamed PyTorch sessions — collectively consuming significant memory. Only the Unsloth Studio container was needed for training. The 30B MoE model at BF16 takes ~63GB, and with batch size already at minimum (1) with `gradient_accumulation=8`, there's no room for waste on 128GB unified memory.

The memory pre-check now catches this automatically before training starts, but the lesson is: always audit running processes before committing GPU memory to a large training run.

### Lessons learned

1. **Smoke-test your toolchain after a gap.** A quick `python -c "import torch; help(torch.dtype)"` or checking the migration guide would have caught the deprecation before wasting a DGX cycle.
2. **Memory pre-check before training is non-negotiable.** On shared or multi-use machines, stale containers and processes silently eat memory. The training script should refuse to start if memory is insufficient — better to fail fast with an actionable message than to OOM mid-training.

---

## 2026-03-28 — Phase 3 complete: model prep scripts, DGX Toolbox integration, and phase restructuring

### Model prep is ready

Phase 3 (Model Prep and Training) is at checkpoint — all scripts written, tested, and integrated with DGX Toolbox. The test suite grew from 46 to 75 tests, all passing:

- `test_prepare_tokenizer.py` — verifies `<wp_gen>` and `<wp_judge>` special tokens are added without duplicates, embeddings are mean-initialized (not zero, not random), and each token resolves to a single ID
- `test_train_model.py` — verifies model download check, LoRA config (r=64, BF16, cosine LR scheduler), `modules_to_save` includes embed_tokens/lm_head, dataset schema (messages format with valid roles), and router logits are enabled for MoE load balancing
- `test_eval_gate.py` — verifies quality gate pass/fail logic against PHPCS pass rate, Spearman correlation, and security thresholds read from config
- `test_eval_gen.py` — verifies PHPCS evaluation runs, security rate detection, and pass rate calculation
- `test_eval_judge.py` — verifies Spearman computation, score inversion detection, and judge output parsing

DGX Toolbox is fully integrated — all training, eval, and serving scripts use the configurable resolver (`scripts/dgx_toolbox.py`) rather than hardcoded paths.

### Decision: Split eval from packaging

**Decision:** Separate evaluation (Phase 4) from packaging and deployment (Phase 5), with a human review checkpoint between them.

**Reasoning:**
- I want to inspect eval results (static eval scores + wp-bench scores) before committing to quantization and release. If results are poor, I need to go back to training — and that's much easier at full BF16 precision than after quantization.
- Quantization is a one-way compression step. AWQ/GGUF can't be reversed to full precision. Keeping eval at full precision means I retain the option to adjust training hyperparameters, add more data, or run additional DPO refinement before packaging.
- The human gate at the end of Phase 4 (plan 04-03) is where I review all eval results and decide whether the model meets the success criteria before proceeding.

### Updated phase structure

| Phase | Name | Status |
|-------|------|--------|
| 1 | Pipeline Ready | Complete |
| 2 | Dataset Production | Complete |
| 3 | Model Prep and Training | At checkpoint (before DGX execution) |
| 4 | Evaluation | Not started (human gate: review results before packaging) |
| 5 | Packaging and Deployment | Not started |

Phase 3 is at checkpoint because the scripts are ready but actual DGX Spark execution (downloading the model, running LoRA fine-tuning) hasn't started yet. That's the next step.

---

## 2026-03-27 — Training strategy: BF16 LoRA (not QLoRA), post-training quantization, and the Unsloth merge bug

### QLoRA is incompatible with MoE models

**Context:** The original memory budget assumed QLoRA (4-bit quantized base + BF16 LoRA adapters) for ~15GB footprint. Research found that Unsloth explicitly states BitsandBytes does not support MoE `nn.Parameter` in 4-bit quantization.

The problem: QLoRA quantizes base model weights to 4-bit NF4. But MoE models have router/gating weights (`nn.Parameter` tensors that decide which experts handle each token) — and BitsandBytes can't quantize these correctly. The result is broken routing where experts don't activate properly.

**Decision:** Use full-precision BF16 LoRA instead. The base model stays in BF16 (~63GB), LoRA adapters train on top in BF16. DGX Spark has 128GB unified memory, so 63GB fits with plenty of headroom for activations and optimizer state. QLoRA would save memory but break the model — and I don't need the savings on this hardware.

### Post-training size reduction path

Quantization is already planned for Phase 4 (no retraining needed):

- **AWQ 4-bit** → ~8GB serving via vLLM (Marlin kernel), minimal quality loss
- **GGUF Q4_K_M** → ~9GB serving via Ollama/llama.cpp, minimal quality loss
- **GGUF Q8_0** → ~16GB serving via Ollama, near-zero quality loss

Since only ~3B params are active per forward pass, inference is already fast even at full precision. Quantization is purely about reducing the serving footprint.

**Future (v2):** Two additional options for further compression:

1. **Knowledge distillation** — use the fine-tuned 30B model as a teacher to train a smaller dense student (Qwen3-8B → AWQ → ~4GB, runs on any consumer GPU). The teacher generates outputs on all training prompts, the student learns from those outputs.

2. **Expert pruning** — analyze which of the 128 experts actually fire on WordPress code (W&B tracks this during training). Remove unused experts and merge similar ones. Could reduce from 128 → 32-64 experts, cutting total params from 30B → 10-15B while keeping the same ~3B active.

### The Unsloth modules_to_save merge bug

**Context:** When fine-tuning with `modules_to_save=["embed_tokens", "lm_head"]`, the special token embeddings (`<wp_gen>`, `<wp_judge>`) are trained as part of the LoRA adapter. The bug (Unsloth GitHub issue #3444): calling `model.merge_and_unload()` followed by save/reload silently dropped `modules_to_save` weights. The merged model contained the original untrained embeddings for the special tokens, not the fine-tuned ones. The model would load but `<wp_gen>` and `<wp_judge>` would produce random outputs.

**Research finding:** The fix is Unsloth-zoo PR #369, merged 2026-01-30, first shipped in unsloth-zoo 2026.2.1. The latest PyPI version is 2026.3.5 (which also includes a follow-up PR #559 for an embed_tokens edge case). DGX Toolbox uses `nvcr.io/nvidia/pytorch:25.11-py3` and installs `unsloth` + `unsloth_zoo` via `pip install --no-deps` at container launch — so it automatically gets the latest fixed version from PyPI.

**Decision:** Our environment is unaffected, so merging should work correctly. However, as defense-in-depth:

1. Save the LoRA adapter separately alongside the tokenizer (don't merge immediately)
2. Verify before merge: merge → save → reload → test that special tokens still produce correct outputs
3. Keep vLLM `--lora-modules` as a fallback (loads adapter at inference time, no merge needed)

If the verification roundtrip fails for any reason, I fall back to adapter-only serving — no risk of shipping a model with corrupted embeddings.

---

## 2026-03-27 — Evaluation strategy: solving Claude-in-the-loop circularity

**Context:** The original eval plan used Claude as a judge to score model outputs. This creates a circularity problem: the training data was curated by Claude, the model was trained on Claude-judged examples, and now Claude would evaluate the result. Any systematic bias in Claude's judgments would be invisible — the eval would confirm the training signal rather than independently validating it.

### The problem

If Claude consistently overrates certain patterns (e.g., verbose docblocks) or underrates others (e.g., terse but correct WordPress idioms), those biases flow into the training data. A Claude-based eval would then reward the same biases in the fine-tuned model's outputs. The eval score would look good, but the model might be learning Claude's preferences rather than genuine WordPress quality.

### Decision: wp-bench + custom eval with no Claude in the loop

**Primary eval — [wp-bench](https://github.com/WordPress/wp-bench):** The canonical WordPress benchmark suite. It uses a real WordPress runtime as the grader — generated code is executed against static checks and runtime assertions. Coverage includes hooks, REST API, security, caching, database queries, and more, which aligns directly with my taxonomy. No LLM in the eval loop.

**Custom judge eval — PHPCS/PHPStan ground truth:** For evaluating the `<wp_judge>` pathway, I compare model scores against deterministic static analysis tools (PHPCS with WordPress-Coding-Standards, PHPStan). These are objective, reproducible, and completely independent of Claude. Judge accuracy is measured by correlation with these ground-truth signals.

**Supplementary — held-out test split:** 597 examples from the dataset's test split, evaluated for PHPCS pass rate on generated code. This is a weaker signal (PHPCS catches style/safety but not all quality dimensions) but provides a quick sanity check during training.

### Why this matters

The eval must be independent of the training signal. wp-bench provides execution-based ground truth (does the code actually work in WordPress?), and PHPCS/PHPStan provide static ground truth (does it conform to standards?). Together they cover both functional correctness and standards compliance without any LLM judgment.

---

## 2026-03-27 — Base model pivot: Qwen3-8B CMoE/ToMoE → Qwen3-30B-A3B

**Context:** The original plan called for converting dense Qwen3-8B into a custom MoE (8 experts, top-2 routing) using either CMoE (arxiv:2502.04416) or ToMoE (arxiv:2501.15316). Before committing to the model setup phase, I evaluated the feasibility of both conversion approaches against the alternative of using Alibaba's existing Qwen3-30B-A3B MoE.

### Options evaluated

**Option A — CMoE (Convert Qwen3-8B → MoE):** Training-free dense-to-MoE conversion. Analytically constructs routers from activation statistics, partitions FFN weights into expert shards. ~5 minutes on a single GPU, zero training cost. However: research paper only (no pip package), unverified on Qwen3's SwiGLU FFN architecture, and no confirmed vLLM/Ollama serving compatibility. Medium-high risk.

**Option B — ToMoE (Convert Qwen3-8B → MoE):** Token-level MoE conversion using routing signals from a calibration dataset. Slightly more community validation than CMoE, and the calibration step means routing quality depends on the data you feed it (WordPress code → WP-aware routing). ~10-30 minutes. Still research code with no stable package and the same serving uncertainty. Medium risk.

**Option C — Qwen3-30B-A3B (Pre-built MoE):** Alibaba's official model. ~30B total params, ~3B active per forward pass, 128 experts with top-8 routing. Zero conversion needed — download and fine-tune. Verified Unsloth support, native vLLM serving, Ollama GGUF available. Fits in 128GB unified memory (60GB BF16, or ~15GB with QLoRA). Low risk.

### The serving reality that killed CMoE/ToMoE

The decisive factor wasn't conversion quality — it was serving. Neither CMoE nor ToMoE has:
- A single published model on HuggingFace
- A standard architecture recognized by AutoModel
- vLLM compatibility
- GGUF/Ollama support
- Any community reports of production deployment

Both produce custom architectures that existing inference tooling cannot load. Building a model that can't be served defeats the purpose of an open-weight project.

### Decision: Pivot to Qwen3-30B-A3B

I'm pivoting to Qwen3-30B-A3B to keep the MoE architecture. The tradeoffs:

- **128 experts is overkill** for two task modes, but task tokens (`<wp_gen>`, `<wp_judge>`) can still influence which experts fire via attention patterns. My hope is that fine-tuning narrows the active expert set per task, even if the full 128 remain available.
- **~3B active params is smaller** than the original 4B target — faster inference than planned.
- **30B total params takes more disk** (60GB BF16 vs 16GB) but fits comfortably on DGX Spark's 128GB unified memory.
- **The "dense-to-MoE conversion" story is gone.** This is no longer a demonstration of CMoE/ToMoE methodology. The project focus shifts entirely to the dataset and fine-tuning quality.
- **Every link in the toolchain is verified:** Unsloth LoRA → vLLM serving → Ollama GGUF → HuggingFace Hub. No unknowns.

The fundamental principle: ship a model people can actually run, rather than demonstrate a conversion technique that produces an unservable artifact.

### Impact on project

- `wp-moe.md` architecture spec needs updating (128 experts top-8 instead of 8 experts top-2)
- README base model table needs updating
- Memory budget for training changes (QLoRA likely required)
- The dataset pipeline is unaffected — training data is model-agnostic

---

## 2026-03-26 — Retrospective: project genesis and architectural choices

### Why this project exists

A search for open-source, open-weight models fine-tuned on WordPress coding best practices turned up nothing. The tools that exist in this space are wrappers — some quite sophisticated — around frontier closed-source models (OpenAI, Claude, etc.). No one had published an open model that internalised WordPress Coding Standards, security patterns, and architectural opinions. So I decided to build one.

The motivation is explicitly open-source: produce a model that the WordPress community can run locally, inspect, modify, and redistribute without vendor lock-in.

### Base model: Qwen3-8B

**Decision:** Use Qwen3-8B as the base model.

**Reasoning:**
- Relatively small (~8B params) — an accessible starting point for experimentation, especially on a single DGX Spark (128GB unified memory).
- Strong out-of-the-box PHP/web code understanding for its size class.
- LLaMA-compatible architecture, making it easy to distribute and adopt. Users can serve it via Ollama, vLLM, llama.cpp, etc. without custom inference code.
- Good HuggingFace ecosystem support and Unsloth compatibility for efficient LoRA fine-tuning.
- The 8B size is a deliberate tradeoff: I sacrifice some raw capability vs. larger models in exchange for fast iteration, low serving cost, and broad accessibility.

### Architecture: MoE with task-token routing

**Decision:** Convert the dense Qwen3-8B into a Mixture-of-Experts model (8 experts, top-2 routing, ~4B active params per forward pass) with two modes:
- `<wp_gen>` — code generation expert pathway
- `<wp_judge>` — structured critique expert pathway

**Reasoning:**
- A single model that both generates and critiques enables a self-improving loop: generate → judge → iterate. This is more practical for end users than managing two separate models.
- MoE keeps active parameter count at ~4B while retaining 8B total capacity, balancing inference speed with model expressiveness.
- First-token routing via special tokens (`<wp_gen>`, `<wp_judge>`) is simple to implement and simple for users to understand. No complex prompt engineering needed — just prepend the task token.
- The goal is an *opinionated* model that pushes back on poor functional or architectural decisions. The judge pathway is central to this: it doesn't just score code, it explains *why* something is wrong and what should be done instead.

This architecture was developed through a combination of research and iterative discussion with Claude, drawing on the LLaMA-MoE methodology for dense-to-sparse conversion.

### Dataset strategy: positive AND negative examples

**Decision:** Curate both high-quality and deliberately poor-quality code examples.

**Reasoning:**
- A model trained only on good code can generate good code but cannot reliably *identify* bad code or explain what makes it bad.
- The judge pathway needs contrastive training data: code that scales poorly, is open to vulnerabilities, creates technical debt, violates WPCS, uses unsafe SQL patterns, etc.
- Three sources of negative examples:
  1. **Real code that fails assessment** — extracted from repos but caught by PHPCS pre-filter or Claude judge.
  2. **Automated mutations** — programmatic degradation of passing code (remove `prepare()`, strip nonces, inject `SELECT *`, etc.). These produce controlled bad→good pairs.
  3. **Synthetic contrastive pairs** — Claude-generated bad→good examples with CoT explanations of the defect and fix.

### Data sourcing: Perplexity Computer agents for repo curation

**Decision:** Used Perplexity computer agents to autonomously gather the top 1000 plugins (by active installs) and top 100 themes from the WordPress ecosystem, along with metadata: GitHub repo URLs, CVSS scores, WordPress.org ratings, update status, tested-up-to WP core versions, and known tags.

**Reasoning:**
- Manual curation of 1100 repositories would be prohibitively slow.
- The metadata (especially CVSS scores and ratings) feeds into quality tiering decisions — a plugin with known CVEs gets different treatment than WordPress Core.
- Automated collection ensures reproducibility: the same criteria can be re-run as the ecosystem evolves.

### Lesson learned: LLM cost estimates are unreliable

**Observation:** Claude initially estimated 35-60 USD in API costs for the Phase 1 judge pipeline. By ~35 repositories processed, actual spend had reached 90 USD (with auto-reload enabled on the Anthropic API billing — a mistake in itself, as it allowed runaway spend without a hard stop).

**Takeaway:**
- **Never trust cost guidance from an LLM.** Models cannot accurately predict their own token consumption across a real pipeline with retries, variable code lengths, and multi-turn judge conversations.
- **Always be conservative.** Set hard billing limits. Disable auto-reload. Run a small pilot batch (5-10 repos) and extrapolate actual per-repo cost before committing to the full corpus.
- **Subsequent pivot:** This cost experience was a factor in the later decision to switch from direct Claude API calls to using Claude Code agents (covered by subscription) for all LLM-driven pipeline work — a change that eliminated per-token API costs entirely for the judge and generation steps.

### Current state (as of this entry)

- 54 repos cloned and PHP functions extracted.
- Phase 1 scripts hardened with utils.py integration.
- Phase 2 scripts (gap analysis, generation, judging, judge dataset) written and hardened.
- Phase 3 (CoT + export) written with 40/60 gen/judge ratio and multi-format export.
- Taxonomy covers 13 categories, 87 tags with minimum coverage targets.
- Pipeline execution has begun but is not yet complete.
- Phases B-E (model setup, training, evaluation, packaging) are planned but not started.

---

## 2026-03-26 — Journal created

Starting this journal to capture design choices, tradeoffs, and lessons learned across the data pipeline and model development phases. Entries are reverse-chronological (newest first).

---

<!-- Template for new entries:

## YYYY-MM-DD — Title

**Context:** What prompted this decision or observation.

**Decision / Observation:** What I chose or noticed.

**Reasoning:** Why — tradeoffs considered, alternatives rejected.

**Outcome:** (fill in later) What actually happened.

-->
