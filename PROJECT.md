# PROJECT: WordPress Best-Practice MoE Model (wp-qwen3-moe)

## Vision

A single Qwen3-based Mixture-of-Experts model that both **generates** and **judges** WordPress code according to strict WordPress Coding Standards. Task tokens (`<wp_gen>`, `<wp_judge>`) route input to specialized expert pathways within the same network. Built and served on the [DGX Toolbox](~/dgx-toolbox) infrastructure stack.

## Architecture

- **Base:** Qwen3-30B-A3B (native MoE, ~30B total params, ~3B active per forward pass, 128 experts, top-8 routing)
- **Modes:** `<wp_gen>` (code generation) and `<wp_judge>` (structured critique with rubric scoring)
- **Compatibility:** HuggingFace `AutoModelForCausalLM`, standard transformers tooling
- **Infrastructure:** DGX Toolbox — Unsloth Studio (fine-tuning), vLLM/Ollama (inference), eval-toolbox (benchmarks), safety harness (guardrails)

See [wp-moe.md](wp-moe.md) for full model specification.

## Execution Model

All LLM-heavy pipeline work uses **Claude Code agents** instead of the Anthropic API:
- **$0 cost** — covered by Claude subscription
- **Parallel execution** — spawn 4-8 agents processing batches simultaneously
- **Spawn-until-target** — continuously spawn agents until data targets are met
- **Quality gates** — same strict rubric (threshold >= 8, security auto-FAIL) applied by agents

See [docs/AGENT_PIPELINE.md](docs/AGENT_PIPELINE.md) for full agent execution model, output format contracts, and scaling guide.

Non-LLM steps (cloning, extraction, gap analysis, mutations, export) run as Python scripts.

---

## Project Phases

### Phase A: Data Pipeline (this repo)

The data pipeline lives in this directory and produces the training dataset.

#### A1. Repository Curation & Extraction

| Step | Script | Description |
|------|--------|-------------|
| A1.1 | *Manual* | Curate plugin/theme list in `config/repos.yaml` |
| A1.2 | `phase1_clone.py` | Shallow-clone all repositories |
| A1.3 | `phase1_extract.py` | Extract functions via PHP tokenizer (`php_extract_functions.php`) |
| A1.4 | `phase1_judge.py` | **PHPCS pre-filter** rejects high-error-density code cheaply; survivors go to **Claude judge** for 9-dimension assessment (WPCS, SQL safety, security, performance, WP API, code quality, dependencies, i18n, accessibility) |

**Quality tiers:**
- `core` — WordPress Core. Auto-passed as reference implementation, tagged only.
- `assessed` — Everything else. Function-by-function pass/fail. No partial credit.

**Outputs:** `data/phase1_extraction/output/passed/` and `data/phase1_extraction/output/failed/`

#### A2. Synthetic Generation & Judge Data

| Step | Script | Description |
|------|--------|-------------|
| A2.1 | `phase2_gap_analysis.py` | Compare tag coverage against `config/taxonomy.yaml` minimums |
| A2.2 | `phase2_mutate.py` | **Automated mutation** of passed real code: remove `prepare()`, strip nonces, strip escaping, remove capability checks, strip sanitization, strip i18n, inject `SELECT *`. Verified detectable by PHPCS. Produces bad->good contrastive pairs. |
| A2.3 | `phase2_generate.py` | Claude generates synthetic code grounded in real Phase 1 style anchors. Fills taxonomy gaps. Contrastive pair templates for Claude-generated bad->good pairs. |
| A2.4 | `phase2_judge.py` | Same judge criteria as Phase 1. Failed synthetics get one revision attempt, then discard. |
| A2.5 | `phase2_judge_dataset.py` | Generates `<wp_judge>` training data: Claude scores passed code (high), failed code (low), and mutated code (controlled defects) on a 0-100 rubric across 6 dimensions. Sanity-checked against expected quality tier. |

**Outputs:**
- `data/phase2_synthetic/output/judged/` — passed/failed synthetic code
- `data/phase2_synthetic/output/mutated/` — automated contrastive pairs
- `data/phase2_synthetic/output/judge_training/` — rubric-scored judge examples

#### A3. Chain-of-Thought & Export

| Step | Script | Description |
|------|--------|-------------|
| A3.1 | `phase3_cot.py` | **Instruction synthesis** for real code (reverse-engineer prompts). **CoT reasoning** for complex examples (SQL, performance, architecture). **Contrastive CoT** for mutation pairs (explain defect + fix). Merges judge training data. |
| A3.2 | `export_dataset.py` | Adds `<wp_gen>`/`<wp_judge>` task tokens. Exports OpenAI JSONL, Alpaca JSON, and raw JSONL with metadata. 80/10/10 train/val/test split. |

**Final outputs in `data/final_dataset/`:**
- `openai_{train,val,test}.jsonl`
- `alpaca_{train,val,test}.json`
- `raw_{train,val,test}.jsonl`
- `metadata.json`

---

### Phase B: Model Setup & Training

*Complete. Implemented in Phases 3 and 6.*

| Step | Description | Status |
|------|-------------|--------|
| B1 | Download Qwen3-30B-A3B base model (native MoE, no conversion needed) | Done |
| B2 | Extend tokenizer with `<wp_gen>` (ID 151669), `<wp_judge>` (ID 151670), mean-initialised embeddings | Done |
| B3 | BF16 LoRA SFT via Unsloth (not QLoRA — MoE router weights incompatible with BitsandBytes 4-bit). LoRA r=32, α=64, targeting q/k/v/o_proj + gate_up/down_proj. | Done |
| B4 | 5 sequential training runs at gen/judge ratios 30/70 through 70/30, each producing isolated adapter in `adapters/qwen3-30b-wp-{ratio}/` | Done |
| B5 | Memory pre-check (70GB minimum), memory watchdog callback, OOM-aware adaptive resource planning between runs | Done |
| B6 | Power-primary adaptive training planner with batch coupling, Unsloth override detection, warmup probes (v1.1) | Done |

### Phase C: Base-Model Profiling & Evaluation (Triage)

*Not yet started. Next milestone step.*

Phase 4 profiles the base model first (gates whether to train 60/40 and 70/30), then evaluates available adapters as a triage gate. Survivors are carried to Phase 7 where fine-tuned adapter routing concentration determines the final ratio selection.

The profiling step computes **routing entropy** and **effective expert count** per MoE layer:

```
H_l = -Σ p_{l,e} · log(p_{l,e})    (routing entropy, layer l, E=128 experts)
E_eff_l = exp(H_l)                   (effective expert count — how many experts are meaningfully active)
```

| Step | Description |
|------|-------------|
| C1 | **Base-model profiling (~minutes):** Gradient-free forward pass with all 5 ratio data distributions, compute E_eff per layer per ratio. If E_eff trends down at 60/40 and 70/30 → start training in background (~2 days); if flat/up → skip. |
| C2 | Serve existing adapters (30/70, 40/60, 50/50) via vLLM, run static eval suite per ratio — in parallel with any background training |
| C3 | wp-bench execution and knowledge tests per ratio |
| C4 | **Triage:** eliminate ratios that fail hard gates or are >5pp behind the best; carry all survivors to Phase 7 (high bar for elimination, low bar for continuation — 1-2pp differences may invert after pruning) |
| C5 | Human review of eval results + E_eff profiling data, approve triage decisions |

### Phase D: MoE-Sieve Selective Training (v2.0)

*Planned. Depends on Phase C (surviving ratios with eval scores + base-model E_eff).*

| Step | Description |
|------|-------------|
| D1 | **Fine-tuned adapter profiling (all survivors):** Gradient-free forward pass per surviving ratio's adapter (not base model — that was C1), per-task-token expert activation counts, E_eff per layer. Compared against base-model E_eff to quantify how LoRA shifted routing. |
| D2 | **Ratio selection gate:** Decision matrix combining Phase C eval score + Phase D1 adapter E_eff (mean/max/variance) — lowest E_eff at equivalent quality (within 2pp) wins; single ratio from here |
| D3 | **MoE-Sieve SFT:** LoRA on hot experts only (+ attention, routers, shared experts), task-aware data filtering, k-sweep at 10%/25%/50% expert budgets |
| D4 | **Comparative eval:** A/B compare each k-sweep adapter against v1.0 full-LoRA on wp-bench and all 9 eval dimensions |

### Phase E: GRPO & Production Deployment (v3.0)

*Planned. Depends on Phase D (MoE-Sieve eval results).*

| Step | Description |
|------|-------------|
| E1 | **Reward infrastructure:** Composite reward pipeline (70% verifiable / 30% frozen judge), security hard gate, MO-GRPO normalisation, VeRPO partial credit |
| E2 | **GRPO training:** Gen-only GRPO on hot experts with RSPO router-shift stabilisation |
| E3 | **LoRA merge:** Bake MoE-Sieve + GRPO adapters into base weights (required before pruning) |
| E4 | **Pruning (AIMER vs REAP):** AIMER (~1 sec, no calibration, task-agnostic baseline) and REAP (~3 hrs, WordPress calibration data, domain-aware) both run at 25%/50%/75% — 6 variants evaluated via gating mask across all 9 dimensions; domain specificity analysis quantifies expert overlap per layer |
| E5 | **Comparative eval:** A/B compare GRPO+pruned model against v2.0 SFT-only on all dimensions + speed delta + model size |
| E6 | **Packaging:** Cascading compression gates (bf16 baseline → quantisation decision → HuggingFace upload → E2E inference validation) |

See [ROADMAP.md](.planning/ROADMAP.md) for full phase details, requirements, and success criteria.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/repos.yaml` | 236 repos (top + poor-quality plugins/themes) with quality tiers, path filters |
| `config/judge_system.md` | Claude judge system instruction (9 dimensions + rubric) |
| `config/taxonomy.yaml` | 87 concept tags + minimum coverage targets per tag |
| `config/synthetic_prompts.yaml` | Generation templates + rejection examples, keyed by gap tag |
| `config/train_config.yaml` | Training hyperparameters (LoRA, scheduler, etc.) |
| `config/adaptive_planning.yaml` | Power-primary adaptive planning thresholds and ladder |
| `config/wp-bench.yaml` | Evaluation benchmark config |
| `config/dgx_toolbox.yaml` | DGX Toolbox execution engine config (container defs, mounts, validation) |

## Directory Structure

```
wp-finetune/
├── PROJECT.md                          # This file
├── README.md                           # Quick start guide
├── JOURNAL.md                          # Engineering decisions log
├── CHANGELOG.md                        # Version history
├── wp-moe.md                           # Full model specification
├── config/
│   ├── repos.yaml                      # 236 repos (top + poor-quality plugins/themes)
│   ├── judge_system.md                 # 9-dimension judge criteria
│   ├── taxonomy.yaml                   # 87 concept tags + coverage minimums
│   ├── synthetic_prompts.yaml          # Generation templates + rejection examples
│   ├── train_config.yaml               # Training hyperparameters (LoRA, scheduler)
│   ├── adaptive_planning.yaml          # Power-primary adaptive planning config
│   ├── wp-bench.yaml                   # Evaluation benchmark config
│   └── dgx_toolbox.yaml               # DGX Toolbox execution engine config
├── scripts/
│   ├── utils.py                        # Shared utilities (JSON parsing, backoff, checkpoints)
│   ├── dgx_toolbox.py                  # Execution engine: validate → resolve → Docker exec
│   ├── adaptive_planner.py             # Power-primary adaptive config engine
│   ├── pipeline_orchestrator.py        # Pipeline state tracker + action planner
│   ├── download_model.py               # Download base model from HuggingFace
│   ├── prepare_tokenizer.py            # Extend tokenizer with <wp_gen>/<wp_judge>
│   ├── train_model.py                  # BF16 LoRA SFT with memory watchdog + OOM recovery
│   ├── merge_adapter.py                # Merge adapter with verification roundtrip
│   ├── phase1_{clone,extract,judge}.py # Phase 1: clone, extract, PHPCS + Claude judge
│   ├── phase2_{gap_analysis,mutate,generate,judge,judge_dataset}.py
│   ├── phase3_cot.py, merge_dataset.py, export_dataset.py
│   └── (+ csv_to_repos, preflight)
├── eval/
│   ├── rubric_definitions.py           # 193 check IDs across 9 weighted dimensions
│   ├── rubric_scorer.py                # 4-tool ground truth scoring engine
│   ├── eval_gen.py                     # Generator eval (9-dimension rubric scoring)
│   ├── eval_judge.py                   # Judge eval (per-dimension Spearman correlation)
│   └── eval_gate.py                    # Quality gate (pass/fail against thresholds)
├── docs/
│   ├── AGENT_PIPELINE.md               # Agent execution model and output format contracts
│   ├── eval/                           # Rubric docs + research backing
│   └── wp-finetune:*.md                # Skill definitions (pipeline, training, observe, etc.)
├── data/
│   ├── phase1_extraction/              # Cloned repos + extracted/passed/failed functions
│   ├── phase2_synthetic/               # Gap reports + synthetic/mutated/judge training data
│   ├── phase3_cot/                     # CoT reasoning checkpoints
│   ├── final_dataset/                  # Train/val/test in OpenAI, Alpaca, Raw JSONL formats
│   └── checkpoints/                    # Pipeline execution checkpoints
├── adapters/                           # LoRA adapter checkpoints per ratio
├── models/                             # Merged model checkpoints
├── telemetry/                          # Training telemetry (thermal logs, agent reports)
└── tests/                              # 75 tests (13 test files)
```

## Dataset Composition (Actual)

267K merged examples (134K judged functions + 143K judge training + 29K CoT), exported at 5 gen/judge ratios after dedup:

| Ratio | Gen | Judge | Total | Train (80%) |
|-------|-----|-------|-------|-------------|
| 30/70 | 13,071 | 30,498 | 43,569 | 34,855 |
| 40/60 | 20,332 | 30,498 | 50,830 | 40,664 |
| 50/50 | 30,498 | 30,498 | 60,996 | 48,796 |
| 60/40 | 45,747 | 30,498 | 76,245 | 60,996 |
| 70/30 | 71,162 | 30,498 | 101,660 | 81,328 |

**Sources:** Top 1000 plugins + top 100 themes (high-quality generation data), 1000 poorly-rated plugins + 186 poorly-rated themes (negative judge data), plus WordPress Core as reference implementation.

**4-way CoT split:** Gen pattern CoT, judge rubric CoT, judge contrastive CoT, shared security CoT — each with max(500, 10%) floor.

## Quality Gates

Every code example in the final dataset passed at least one of:
1. **WordPress Core origin** (auto-passed as reference implementation)
2. **PHPCS pre-filter** (< 5 errors per 100 lines) **AND** Claude 9-dimension judge (threshold >= 8/10 per dimension, security auto-FAIL below 5)
3. **Claude synthetic generation** + same judge criteria (with one revision attempt on failure)

Judge training data is additionally sanity-checked: high-quality source code must score > 50 overall, low-quality must score < 95.

## Success Criteria

| Metric | Target | Measured At |
|--------|--------|-------------|
| Generator PHPCS pass rate | > 95% | Phase C (Evaluation) |
| Generator security pass rate | > 98% | Phase C (Evaluation) |
| Judge Spearman correlation | > 0.85 | Phase C (Evaluation) |
| Judge classification precision | > 0.90 | Phase C (Evaluation) |
| Active parameters per inference | ~3B (top-8 of 128 experts) | Phase B (confirmed) |
| Inference latency (DGX Spark) | < 2s via vLLM | Phase E (Packaging) |

## Current Status

- [x] Phase A: Data pipeline — 267K examples, 5 ratio exports (v1.0 Phases 1-2)
- [x] Phase B: Model setup & training — 5 LoRA runs complete on DGX Spark (v1.0 Phase 3)
- [x] Adaptive training infrastructure — power-primary planner, memory watchdog (v1.1 Phase 6)
- [ ] Phase C: Base-model profiling + evaluation triage — **next step** (v1.0 Phase 4)
- [ ] Phase D: MoE-Sieve selective training (v2.0 Phases 7-9)
- [ ] Phase E: GRPO + pruning + packaging (v3.0 Phases 10-14)

## Dependencies

**Runtime:**
- Python 3.10+
- `pyyaml`, `python-dotenv`
- PHP CLI with `tokenizer` extension
- PHP_CodeSniffer + WordPress-Coding-Standards
- [Claude Code](https://claude.com/claude-code) (subscription) — all LLM pipeline steps run via agents

**Training & Eval — via DGX Toolbox:**
- Unsloth Studio — BF16 LoRA fine-tuning (not QLoRA — MoE router incompatibility)
- eval-toolbox container — lm-eval benchmarks, MLflow tracking
- vLLM (:8020) — model serving for eval + production inference
- LiteLLM (:4000) — unified API for cross-model evaluation
- Ollama (:11434) — GGUF quantised local serving
- Open-WebUI (:12000) — interactive demo/testing
- Hardware: DGX Spark (Blackwell GB10, 128GB unified memory)
