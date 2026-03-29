# WordPress Best-Practice MoE Model: Project Specification

**Project Goal:** Fine-tune Qwen3-30B-A3B into an opinionated WordPress code model that generates and judges PHP code against strict WordPress Coding Standards, using task-token routing (`<wp_gen>`, `<wp_judge>`) within a single MoE network. Built and served entirely on the [DGX Toolbox](https://github.com/dr-robert-li/dgx-toolbox) stack.

**Base Model:** Qwen3-30B-A3B (native MoE — no dense-to-MoE conversion needed)
**Infrastructure:** DGX Spark (Blackwell GB10, 128GB unified memory) via DGX Toolbox
**Primary Use Cases:**
- Code generation: `<wp_gen>` → Production-ready WordPress plugin/theme code
- Code judging: `<wp_judge>` → Structured critique with 9-dimension rubric scoring

---

## 1. Foundation Model

### Selected Base Model

**Qwen3-30B-A3B** (`Qwen/Qwen3-30B-A3B`)

| Property | Value |
|----------|-------|
| Total params | ~30B |
| Active params | ~3B per forward pass |
| Experts | 128 experts, top-8 routing per token |
| Architecture | `Qwen2MoeForCausalLM` |
| Training | BF16 LoRA via Unsloth (not QLoRA — MoE router weights incompatible with BitsandBytes 4-bit) |
| Memory footprint | ~63GB BF16 (fits DGX Spark's 128GB with headroom for optimizer states + activations) |
| Serving | vLLM (native), Ollama (GGUF), HuggingFace (AutoModel) — all verified |

### Why Qwen3-30B-A3B over dense-to-MoE conversion

CMoE (arxiv:2502.04416) and ToMoE (arxiv:2501.15316) were evaluated for converting dense Qwen3-8B into a custom MoE. Both were rejected: neither has published models on HuggingFace, neither produces architectures recognised by vLLM or llama.cpp, and neither has community reports of production deployment. Using Qwen3-30B-A3B trades the "dense-to-MoE conversion" narrative for a proven, servable model with zero conversion risk.

### Tokenizer Extensions

Two special tokens added to the Qwen3 tokenizer:
- `<wp_gen>` (ID 151669) — generation mode
- `<wp_judge>` (ID 151670) — evaluation mode

Embeddings are mean-initialised from existing vocabulary (not zero, not random). Both tokens resolve to single IDs. The tokenizer is saved separately at `adapters/tokenizer/` and verified before merge.

### DGX Toolbox Integration

All container management is handled by `scripts/dgx_toolbox.py` — a config-driven execution engine that reads `config/dgx_toolbox.yaml` for paths, container definitions, validation checks, and pinned dependency versions. No hardcoded Docker commands.

| Container | Purpose |
|-----------|---------|
| **Unsloth Studio** (:8000) | Model download, tokenizer extension, LoRA training, adapter merge |
| **eval-toolbox** | Evaluation suite (PHPCS, PHPStan, Spearman, wp-bench), W&B tracking |
| **vLLM** (:8020) | Model serving for eval and production inference |
| **LiteLLM** (:4000) | Unified API proxy for cross-model evaluation |
| **Ollama** (:11434) | GGUF quantised local serving |
| **Open-WebUI** (:12000) | Interactive demo/testing interface |

---

## 2. Training Data

### Sources

| Dataset | Entries | GitHub URLs | Purpose |
|---------|---------|-------------|---------|
| Top 1000 plugins (by installs) | 1,000 | 776 (77.6%) | High-quality generation examples |
| Top 100 themes (by installs) | 100 | 25 (25%) | High-quality generation examples |
| Poor plugins (<=3 stars, 100+ installs) | 1,000 | 163 (16.3%) | Negative judge examples |
| Poor themes (<=3 stars, 100+ installs) | 186 | 1 (0.5%) | Negative judge examples |
| WordPress Core | 1 | 1 | Auto-passed reference implementation |

**Total repos in `config/repos.yaml`:** 236

GitHub URLs discovered via 3-phase process: WordPress.org page scraping (356 repos), `gh search repos` CLI search (501 repos), validation pass to classify official vs mirror repos and remove false positives.

### Pipeline

All LLM-heavy steps run via Claude Code agents (covered by subscription, $0 API cost). Non-LLM steps run as Python scripts.

```
Clone repos → Extract PHP functions → PHPCS pre-filter → Agent judge (9-dimension rubric)
    → passed/failed split → gap analysis → synthetic generation → judge synthetics
    → judge training data (score all passed + failed) → 4-way CoT → merge → export
```

**Quality gates:** Every non-core example passes PHPCS pre-filtering AND 9-dimension rubric assessment (threshold >= 8/10 per dimension, security auto-FAIL below 5). WordPress Core functions are auto-passed.

### Dataset Composition

267K merged examples (134K judged functions + 143K judge training + 29K CoT), deduplicated and exported at 5 gen/judge ratios:

| Ratio | Gen | Judge | Total | Train (80%) |
|-------|-----|-------|-------|-------------|
| 30/70 | 13,071 | 30,498 | 43,569 | 34,855 |
| 40/60 | 20,332 | 30,498 | 50,830 | 40,664 |
| 50/50 | 30,498 | 30,498 | 60,996 | 48,796 |
| 60/40 | 45,747 | 30,498 | 76,245 | 60,996 |
| 70/30 | 71,162 | 30,498 | 101,660 | 81,328 |

Split: 80/10/10 train/val/test per ratio.

### 4-Way CoT Split

| Type | Source | Teaches | Min floor |
|------|--------|---------|-----------|
| Gen: Pattern CoT | Passed functions | "Requirement → pattern → implementation → reasoning" | max(500, 10%) |
| Judge: Rubric CoT | Mixed passed+failed | "Code → walk through 9 dimensions → scores → verdict" | max(500, 10%) |
| Judge: Contrastive CoT | Failed functions | "Bad code → issues → fixes → what good version looks like" | max(500, 10%) |
| Shared: Security CoT | Security-tagged functions | "Security analysis → nonce/cap/escape → verdict" | max(500, 10%) |

### Data Format

All training data uses OpenAI messages format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a WordPress code expert..."},
    {"role": "user", "content": "<wp_gen> Create a REST API endpoint..."},
    {"role": "assistant", "content": "<?php\nfunction wpgen_register_routes() {..."}
  ]
}
```

---

## 3. Training Strategy

### BF16 LoRA (not QLoRA)

QLoRA quantises base model weights to 4-bit NF4, but MoE models have router/gating weights (`nn.Parameter` tensors) that BitsandBytes can't quantise correctly. The result would be broken expert routing. BF16 LoRA keeps the full-precision base (~63GB) with adapters trained on top.

### Training Configuration

```yaml
model: Qwen/Qwen3-30B-A3B
max_seq_length: 4096

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_up_proj, down_proj]
  modules_to_save: [embed_tokens, lm_head]

training:
  epochs: 2
  batch_size: 1
  gradient_accumulation_steps: 8    # effective batch size = 8
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.05
  bf16: true
  logging_steps: 10
  eval_steps: 100
  save_steps: 200
  report_to: wandb (project: wp-qwen3-moe)
```

### Loss Function

- Standard cross-entropy on next-token prediction
- MoE auxiliary loss: router z-loss to prevent routing collapse (`output_router_logits=True`)
- Load balancing loss encourages even expert usage

### Multi-Ratio Training

Five LoRA training runs with the same base model and hyperparameters, only the dataset ratio changes (30/70 through 70/30). Each run gets an isolated checkpoint directory. This is a clean A/B/C/D/E test to determine the optimal gen/judge balance empirically.

### Idempotency

Every training step is safe to re-run:

| Step | Skip condition |
|------|---------------|
| Download model | Safetensors shards exist |
| Extend tokenizer | `adapters/tokenizer/` has special tokens |
| Train model | `adapter_config.json` exists (use `--resume` for partial) |
| Merge adapter | Merged model passes token verification |

### Memory Pre-Check

Before loading the 63GB model, `train_model.py` reads `/proc/meminfo` for available RAM (70GB minimum required). If insufficient: shows top memory consumers (processes + Docker containers), prints actionable suggestions, and blocks training (`exit 1`).

### Adapter Merge

The Unsloth-zoo `modules_to_save` merge bug (issue #3444, fixed in PR #369, shipped in unsloth-zoo 2026.2.1) is confirmed fixed in the DGX Toolbox container. As defense-in-depth:

1. Save LoRA adapter separately (don't merge immediately)
2. Verify before merge: merge → save → reload → test that `<wp_gen>` and `<wp_judge>` still produce correct outputs
3. Fallback: vLLM `--lora-modules` loads adapter at inference time without merging

---

## 4. Evaluation

### Eval Circularity Problem

The training data was curated by Claude (judging, CoT generation). Using Claude to evaluate the model's outputs would create a feedback loop — the eval would confirm the training signal rather than independently validating it. All evaluation uses Claude-free ground truth.

### Canonical Rubric

`docs/eval/wp_code_quality_rubric.md` defines 193 check IDs (83 positive signals, 110 negative signals) across 9 weighted dimensions:

| Dimension | Weight | Key Checks |
|-----------|--------|------------|
| WPCS Compliance | 10% | Naming conventions, Yoda conditions, braces, spacing |
| Security | 20% | Nonces, capability checks, escaping, sanitisation |
| SQL Safety | 15% | `$wpdb->prepare()`, no concatenation, parameterised queries |
| Performance | 10% | Caching, N+1 detection, query optimisation |
| WP API Usage | 10% | WP_Query over raw SQL, proper hook usage |
| i18n / l10n | 10% | Translation functions, text domains |
| Accessibility | 8% | ARIA, form labels, screen reader text |
| Error Handling | 10% | WP_Error checks, type safety, graceful failure |
| Code Structure | 7% | Hook patterns, activation lifecycle, REST API patterns |

**Critical floor rules:** Security, SQL Safety, and Code Structure dimensions have automatic score caps. If a direct XSS vector is found, Security cannot score above 3/10 regardless of other patterns.

### Ground Truth Scoring Pipeline

The rubric maps to a 4-tool automated scoring pipeline (no LLM in the loop):

1. **PHPCS** with WordPress, WordPressVIPMinimum, and Security standards (~120 check IDs)
2. **PHPStan** level 5 with wordpress-stubs (type errors, undefined calls, wrong return types)
3. **Regex patterns** for 30+ checks not covered by PHPCS (N+1 loops, missing transient caching)
4. **LLM judgment** for 18 checks requiring semantic understanding (architectural appropriateness, ARIA completeness) — uses a different model than Claude to avoid circularity

### Eval Scripts

| Script | Measures | Against |
|--------|----------|---------|
| `eval/eval_gen.py` | Generator quality | Full 9-dimension rubric scoring on generated code |
| `eval/eval_judge.py` | Judge accuracy | Per-dimension Spearman correlation against ground truth |
| `eval/eval_gate.py` | Quality gate | Pass/fail against configured thresholds |
| `eval/rubric_scorer.py` | Ground truth | 4-tool scoring engine |
| `eval/rubric_definitions.py` | Check registry | All 193 check IDs with weights and detection methods |

### wp-bench

[wp-bench](https://github.com/WordPress/wp-bench) provides execution-based evaluation — generated code is run against a real WordPress runtime with static checks and runtime assertions. Covers hooks, REST API, security, caching, database queries. No LLM in the eval loop.

### Success Criteria

| Metric | Target |
|--------|--------|
| Generator PHPCS pass rate | > 95% |
| Generator security pass rate | > 98% |
| Judge Spearman correlation (overall) | > 0.85 |
| Judge per-dimension correlation (Security, SQL) | > 0.75 |
| Judge classification precision | > 0.90 |
| Overall mean score target | > 75.0 |
| Active parameters per inference | ~3B |

### Multi-Ratio Experiment

1. Export at 30/70, 40/60, 50/50, 60/40, 70/30
2. Train on each (LoRA is cheap enough to run 5 times)
3. Eval each against the canonical rubric via `eval_gen.py`, `eval_judge.py`, `eval_gate.py`
4. Let the data decide the optimal ratio

---

## 5. Post-Training Quantisation & Serving

### Quantisation Formats

| Format | Size | Serving | Quality Loss |
|--------|------|---------|-------------|
| BF16 (training output) | ~60GB | vLLM | None |
| AWQ 4-bit | ~8GB | vLLM (Marlin kernel) | Minimal |
| GGUF Q4_K_M | ~9GB | Ollama / llama.cpp | Minimal |
| GGUF Q8_0 | ~16GB | Ollama | Near-zero |

Quantisation happens after evaluation passes — at full BF16 precision, the model can be adjusted if eval fails. Quantisation is a one-way compression step.

### Serving Stack (DGX Toolbox)

| Component | Port | Purpose |
|-----------|------|---------|
| vLLM | 8020 | High-throughput OpenAI-compatible inference |
| LiteLLM | 4000 | Unified API proxy routing to vLLM/Ollama |
| Ollama | 11434 | GGUF quantised local serving |
| Open-WebUI | 12000 | Interactive demo/testing interface |
| Safety harness | 5000 | Guardrails, rate limiting, PII redaction |

### Future: Knowledge Distillation

After v1 release, the fine-tuned 30B model can serve as a teacher to train smaller dense students:

| Student | Active Params | Serving (AWQ) |
|---------|---------------|---------------|
| Qwen3-8B (dense) | 8B | ~4GB |
| Qwen3-4B (dense) | 4B | ~2GB |
| Qwen3-1.7B (dense) | 1.7B | ~1GB |

### Future: Expert Pruning

Analyse which of the 128 experts fire on WordPress code during fine-tuning (W&B tracks this). Remove unused experts and merge similar ones. Could reduce from 128 → 32-64 experts, cutting total params from 30B → 10-15B while keeping ~3B active.

---

## 6. Distribution

### HuggingFace Hub

```yaml
license: apache-2.0
base_model: Qwen/Qwen3-30B-A3B
tags: [code, wordpress, php, mixture-of-experts, code-generation, code-review]
```

### Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "your-org/wp-qwen3-moe-30b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Generation
prompt = "<wp_gen> Create a settings page for a newsletter plugin with email validation"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)

# Judging
code = "<?php\nfunction display_comment() {\n    echo $_POST['comment'];\n}\n"
judge_prompt = f"<wp_judge> Evaluate this WordPress code:\n\n{code}"
inputs = tokenizer(judge_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
```

### Deployment via DGX Toolbox

```bash
# Serve via vLLM (high throughput)
~/dgx-toolbox/inference/start-vllm.sh

# Serve via Ollama (GGUF quantised)
ollama run wp-qwen3-moe-30b

# Route through LiteLLM for unified API
~/dgx-toolbox/inference/start-litellm.sh

# Interactive demo
~/dgx-toolbox/containers/start-open-webui.sh
```

---

## 7. Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|-----------|
| MoE routing collapse (all experts learn the same thing) | Monitor per-expert activation patterns in W&B; task tokens should create separation |
| 128 experts insufficient for 2-mode separation | Multi-ratio experiment tests whether more judge data improves discrimination |
| Training OOM on 128GB | Memory pre-check blocks if < 70GB available; batch_size=1 with grad_accum=8 |
| Adapter merge corrupts special token embeddings | Save adapter separately; verify before merge; vLLM adapter-loading fallback |
| QLoRA breaks MoE routing | Using BF16 LoRA instead (confirmed incompatibility with BitsandBytes) |

### Quality Risks

| Risk | Mitigation |
|------|-----------|
| Eval circularity (Claude trained data, Claude evaluates) | wp-bench + PHPCS/PHPStan ground truth — no Claude in eval loop |
| Model generates insecure code | Security dimension weighted 20%, critical floor rules, adversarial testing |
| Judge misses critical issues | Per-dimension Spearman correlation with multi-tool ground truth |
| Overfitting to synthetic data | Real code from 236 repos forms the majority; synthetic is gap-fill only |

### Adoption Risks

| Risk | Mitigation |
|------|-----------|
| Model too large for users | AWQ (~8GB) and GGUF (~9GB) quantised versions; future distillation to 4-8B |
| Hard to use | Task tokens are simple: prepend `<wp_gen>` or `<wp_judge>` |
| License concerns | Apache 2.0, compatible with Qwen3 and WordPress GPL |

---

## 8. Success Criteria

**Quantitative**
- Generator PHPCS pass rate: > 95%
- Generator security pass rate: > 98%
- Judge Spearman correlation: > 0.85
- Judge classification precision: > 0.90
- Active parameters per inference: ~3B

**Qualitative**
- Generates idiomatic WordPress code that follows WPCS
- Catches critical security issues reliably (XSS, SQL injection, CSRF, missing capabilities)
- Judge explanations are actionable — not just scores, but what to fix and why
- The model is deliberately opinionated — it pushes back on poor architectural decisions

---

## Appendix A: 9-Dimension Rubric Reference

Each dimension scores 0-10. Overall score is a weighted sum normalised to 0-100:

```
overall = (D1×10 + D2×20 + D3×15 + D4×10 + D5×10 + D6×10 + D7×8 + D8×10 + D9×7) / 10
```

**Dimension 1: WPCS Compliance (10%)** — Naming conventions, Yoda conditions, control structure spacing, global prefixing, file naming, strict comparisons.

**Dimension 2: Security (20%)** — Nonce verification, capability checks, output escaping (`esc_html`, `esc_attr`, `esc_url`, `wp_kses`), input sanitisation (`sanitize_text_field`, `absint`), CSRF protection, direct file access prevention. **Critical floor: unescaped user output caps score at 3/10.**

**Dimension 3: SQL Safety (15%)** — `$wpdb->prepare()` for all parameterised queries, no string concatenation in SQL, use of WP_Query/get_posts over raw SQL where possible, proper table prefixing. **Critical floor: direct variable interpolation in SQL caps score at 2/10.**

**Dimension 4: Performance (10%)** — Object/transient caching, `no_found_rows` and `fields => 'ids'` optimisations, N+1 query detection, autoload option management, batch operations.

**Dimension 5: WP API Usage (10%)** — `WP_Query` over raw SQL, `wp_remote_get` over `curl`, proper hook registration (actions/filters), `wp_enqueue_script/style` over direct `<script>` tags.

**Dimension 6: i18n / l10n (10%)** — All user-facing strings wrapped in `__()`, `_e()`, `esc_html__()`, etc. Correct text domain usage. Pluralisation via `_n()`.

**Dimension 7: Accessibility (8%)** — Semantic HTML, ARIA attributes, form labels, screen reader text, keyboard navigation, focus management.

**Dimension 8: Error Handling (10%)** — `WP_Error` checks on API calls, `is_wp_error()` guards, type checking, graceful failure with user-facing messages, no silent failures.

**Dimension 9: Code Structure (7%)** — Hook-based architecture (not procedural), proper activation/deactivation hooks, REST API patterns (permission callbacks, schema definitions), single responsibility.

---

## Document Version
- Version: 2.0
- Date: March 2026
- Status: Phase 3 at checkpoint — training scripts ready, awaiting DGX execution
