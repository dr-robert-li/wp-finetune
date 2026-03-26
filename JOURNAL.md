# Engineering Journal — wp-qwen3-moe

Decisions, reasoning, and observations logged as the project evolves.

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
