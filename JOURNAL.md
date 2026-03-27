# Engineering Journal — wp-qwen3-moe

Decisions, reasoning, and observations logged as the project evolves.

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
