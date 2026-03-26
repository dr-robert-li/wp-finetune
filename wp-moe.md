# WordPress Best-Practice MoE Model: Project Specification

**Project Goal:** Fine-tune a Qwen3-30B-A3B-based Mixture-of-Experts (MoE) model that generates and judges WordPress code according to strict WordPress Coding Standards (WPCS), combining both capabilities in a single network using task tokens. Built and served entirely on the [DGX Toolbox](~/dgx-toolbox) stack.

**Target Architecture:** Single shared LLM with task-token routing + sparse MoE layers
**Base Model Strategy:** Qwen3-30B-A3B native MoE (no conversion needed)
**Infrastructure:** DGX Toolbox (Unsloth Studio, vLLM, eval-toolbox, safety harness)
**Primary Use Cases:**
- Code generation: `<wp_gen>` → Production-ready WordPress plugin/theme code
- Code judging: `<wp_judge>` → Structured critique with WPCS compliance scoring

---

## 1. Foundation Model Selection

### Base Model Requirements
- **Architecture:** Qwen3-based transformer, converted to MoE
- **Size:** ~30B total parameters (~3B active per forward pass, 128 experts, top-8 routing)
- **Code capability:** Qwen3-30B-A3B has strong PHP/web code understanding out of the box
- **HF compatibility:** Must work with `AutoModelForCausalLM` and standard transformers tooling
- **Infrastructure:** Must run on DGX Spark (Blackwell GB10, 128GB unified memory) via DGX Toolbox

### Selected Base Model

**Qwen3-30B-A3B** (`Qwen/Qwen3-30B-A3B`)
- State-of-the-art code generation for its size class, strong PHP/WordPress understanding
- Native MoE architecture (128 experts, top-8 routing, ~3B active params per forward pass)
- No dense-to-MoE conversion needed — production-ready serving via vLLM, Ollama, HuggingFace
- Native HuggingFace integration, Unsloth-compatible for efficient LoRA fine-tuning
- Fits comfortably in DGX Spark's 128GB unified memory for training and inference
- Conversion script partitions dense FFN weights into expert shards + trains gating network

### DGX Toolbox Integration for Model Setup

| Component | Role |
|-----------|------|
| **Unsloth Studio** (:8000) | Interactive fine-tuning with LoRA/QLoRA, manages training sessions |
| **data-toolbox** | Dataset curation, deduplication (datatrove, cleanlab), quality filtering |
| **eval-toolbox** | lm-eval benchmarks, W&B experiment tracking, metric computation |
| **vLLM** (:8020) | High-throughput batch inference for DPO candidate generation |
| **LiteLLM** (:4000) | Unified API routing for cross-model evaluation (Claude, GPT-4, local models) |
| **Ollama** (:11434) | Local model serving for GGUF quantized versions |
| **Open-WebUI** (:12000) | Interactive demo/testing interface |
| **Label Studio** (:8081) / **Argilla** (:6900) | Human annotation for DPO preference data |
| **Safety harness** (:5000) | Guardrails, red-teaming, PII redaction, constitutional AI critique |
| **Triton/TensorRT-LLM** | Production-grade optimized inference |

### Tokenizer Extensions
- Base: Use existing Qwen3 tokenizer
- **Add special tokens:**
  - `<wp_gen>` (generation mode)
  - `<wp_judge>` (evaluation mode)
  - Optional: `<wp_security>`, `<wp_performance>` for finer-grained control
- Implementation: `tokenizer.add_special_tokens({'additional_special_tokens': ['<wp_gen>', '<wp_judge>']})`
- Resize model embeddings: `model.resize_token_embeddings(len(tokenizer))`

---

## 2. Data Requirements & Sources

### 2.1 Positive Code Corpus (Best Practice Examples)

**Goal:** 5,000-10,000 high-quality WordPress code samples that pass WPCS

#### Primary Sources

**WordPress Core & Official Plugins**
- WordPress Core repository: `github.com/WordPress/WordPress`
- Official plugins: WooCommerce, Jetpack, Akismet, Yoast SEO
- **Extraction:** Clone repos, filter `.php` files, run PHPCS to verify compliance
- **Expected yield:** ~3,000 verified samples

**High-Quality Theme/Plugin Repositories**
- WordPress.org reviewed plugins/themes
- Underscores starter theme (`_s`)
- Genesis Framework samples
- **Filter criteria:** 4.5+ rating, 10k+ installs, recent updates
- **Expected yield:** ~2,000 samples

**WordPress Coding Standards Test Suite**
- `github.com/WordPress/WordPress-Coding-Standards`
- Contains canonical examples of correct vs incorrect patterns
- **Expected yield:** ~500 reference examples

#### Data Fields per Sample
```json
{
  "source": "wordpress-core|plugin|theme",
  "file_path": "wp-includes/class-wp-query.php",
  "function_or_class": "WP_Query::parse_query",
  "code_snippet": "...",
  "context": "Query parsing and validation",
  "phpcs_result": {
    "passed": true,
    "standard": "WordPress-Core",
    "errors": 0,
    "warnings": 0
  },
  "complexity_score": "medium",
  "features": ["hooks", "sanitization", "i18n"]
}
```

### 2.2 Contrastive Examples (Bad → Good Pairs)

**Goal:** 3,000-5,000 pairs showing violations and corrections

#### Sources

**PHPCS Test Files**
- WordPress-Coding-Standards test fixtures
- Contains intentional violations with expected fixes
- **Extraction:** Parse test files, extract before/after pairs
- **Expected yield:** ~800 pairs

**Common WordPress Vulnerabilities**
- OWASP WordPress Security documentation
- Plugin vulnerability databases (WPScan, Wordfence)
- **Categories:**
  - SQL injection (missing `$wpdb->prepare`)
  - XSS (missing `esc_html`, `esc_attr`)
  - CSRF (missing nonces)
  - Direct file access (missing `ABSPATH` check)
- **Expected yield:** ~1,000 security pairs

**Automated Mutation**
- Take verified good code
- Introduce controlled violations:
  - Remove sanitization functions
  - Add direct `$_POST`/`$_GET` access
  - Break naming conventions
  - Remove i18n functions
  - Introduce SQL injection patterns
- Run PHPCS to confirm detection
- **Expected yield:** ~2,000 synthetic pairs

#### Data Fields per Pair
```json
{
  "bad_code": "...",
  "good_code": "...",
  "violation_type": "security|wpcs|performance|i18n",
  "specific_issue": "missing-nonce",
  "severity": "critical|high|medium|low",
  "phpcs_errors": ["WordPress.Security.NonceVerification.Missing"],
  "explanation": "Forms must verify nonces to prevent CSRF attacks"
}
```

### 2.3 Synthetic Generation Tasks

**Goal:** 2,000-3,000 instruction→code pairs for diverse WordPress development tasks

#### Task Categories (with quotas)

**Custom Post Types & Taxonomies** (400 samples)
- Varied: public/private, hierarchical/flat, different capabilities
- Templates: `register_post_type`, `register_taxonomy`, archive templates

**Settings Pages & Options** (400 samples)
- Settings API usage, sections, fields
- Sanitization callbacks, capabilities checks
- Different UI patterns: tabs, accordions

**Shortcodes** (300 samples)
- Parameter handling, content wrapping
- Escaping output, various complexity levels

**REST API Endpoints** (300 samples)
- Custom endpoints, permissions callbacks
- Schema definitions, sanitization/validation

**Meta Boxes** (300 samples)
- Different post types, save callbacks
- Nonce verification, capability checks

**Gutenberg Blocks** (400 samples)
- Block registration (PHP side)
- Attributes, render callbacks
- Dynamic vs static blocks

**WooCommerce Extensions** (300 samples)
- Product types, payment gateways
- Hooks and filters usage

**Multisite Functions** (200 samples)
- Network-wide operations
- Site switching, blog-specific queries

**Admin Notices & Ajax** (200 samples)
- Dismissible notices, ajax handlers
- Nonce verification in ajax

**Cron Jobs & Background Tasks** (200 samples)
- wp_schedule_event usage
- Action callbacks, error handling

#### Generation Pipeline

**Stage 1: Task Description Generation**
- Use Claude (via LiteLLM) to generate diverse task descriptions
- Template: "Create a WordPress [component] that [specific requirement] with [constraints]"
- Example: "Create a WordPress settings page for a newsletter plugin that includes email validation, AJAX save, and capability checking for 'manage_options'"

**Stage 2: Code Generation (Multi-Model)**
- Generate 3-5 candidate implementations via LiteLLM unified API:
  - Claude Sonnet/Opus
  - GPT-4
  - Local models via Ollama/vLLM
- Each uses system prompt enforcing WPCS

**Stage 3: Automated Judging**
- Run PHPCS with WordPress-Extra standard
- Parse errors/warnings
- Static analysis checks:
  - Nonce usage in forms
  - Sanitization before DB queries
  - Escaping on output
  - Capability checks before privileged operations

**Stage 4: Manual Review (Sample)**
- Human review 20% of synthetic pairs
- Verify quality and educational value
- Check for unrealistic patterns

**Stage 5: Augmentation**
- For accepted code, create variations:
  - Different parameter names
  - Alternative valid approaches
  - Various complexity levels

#### Synthetic Data Fields
```json
{
  "task_description": "Create a meta box for storing product SKU",
  "instruction": "<wp_gen> Create a WordPress meta box...",
  "generated_code": "...",
  "phpcs_passed": true,
  "validation_checks": {
    "has_nonce": true,
    "has_capability_check": true,
    "sanitizes_input": true,
    "escapes_output": true
  },
  "generation_model": "gpt-4",
  "reviewed": true
}
```

### 2.4 Judge Training Dataset

**Goal:** 3,000-5,000 (code, rubric_scores) pairs

#### Score Dimensions

```json
{
  "overall_score": 0-100,
  "wpcs_compliance": 0-100,
  "security_score": 0-100,
  "performance_score": 0-100,
  "i18n_score": 0-100,
  "accessibility_score": 0-100,
  "documentation_score": 0-100,
  "must_fix_issues": ["array of critical problems"],
  "suggested_improvements": ["array of recommendations"],
  "passes_threshold": true/false
}
```

#### Generation Approach

**Automated Scoring**
- PHPCS provides base scores for WPCS compliance
- Security checklist provides security score
- Custom scripts check for:
  - Translation function usage (i18n)
  - Accessibility attributes (aria-*, alt text)
  - PHPDoc coverage (documentation)
  - Query optimization patterns (performance)

**Human Annotation (Gold Set)**
- Manually score 500 diverse examples
- Use as validation set for judge model
- Establishes ground truth for complex cases

**Hybrid Approach**
- Start with automated scores
- Use Claude/GPT-4 as "silver annotator" (via LiteLLM unified API):
  - Provide rubric and code
  - Generate detailed critique
  - Extract scores and issues
- Human review 20% to validate

**Data Augmentation**
- Take good code (score 90+)
- Introduce controlled defects
- Expected score should decrease predictably
- Verify with PHPCS/security checks

---

## 3. Data Preparation Pipeline

### 3.1 Extraction Phase

**WordPress Core & Plugins**
```bash
# Clone and filter
git clone --depth 1 https://github.com/WordPress/WordPress.git
find WordPress -name "*.php" -type f > php_files.txt

# Run PHPCS on each file
while read file; do
  phpcs --standard=WordPress-Core \
       --report=json \
       "$file" > "results/${file//\//_}.json"
done < php_files.txt

# Parse results, extract passing functions/classes
python extract_compliant_code.py results/ > clean_corpus.jsonl
```

**Tools Required:**
- PHP_CodeSniffer + WordPress-Coding-Standards
- Custom parser scripts (Python)
- Git for repository management

### 3.2 Judgment & Categorization Phase

**PHPCS Integration**
```python
# Pseudo-code
def judge_code(code_snippet, standard="WordPress-Extra"):
    # Write to temp file
    temp_file = write_temp_php(code_snippet)
    
    # Run PHPCS
    result = subprocess.run([
        'phpcs',
        f'--standard={standard}',
        '--report=json',
        temp_file
    ], capture_output=True)
    
    phpcs_data = json.loads(result.stdout)
    
    return {
        'passed': phpcs_data['totals']['errors'] == 0,
        'errors': phpcs_data['totals']['errors'],
        'warnings': phpcs_data['totals']['warnings'],
        'violations': extract_violations(phpcs_data)
    }
```

**Security Validation**
```python
def check_security(code):
    checks = {
        'has_nonce_verification': 
            'wp_verify_nonce' in code or 'check_ajax_referer' in code,
        'has_capability_check': 
            'current_user_can' in code,
        'uses_wpdb_prepare': 
            '$wpdb->prepare' in code if '$wpdb' in code else None,
        'escapes_output': 
            any(esc in code for esc in ['esc_html', 'esc_attr', 'esc_url']),
        'sanitizes_input':
            any(san in code for san in ['sanitize_text_field', 'wp_kses']),
        'prevents_direct_access':
            'ABSPATH' in code or "defined( 'ABSPATH' )" in code
    }
    return checks
```

**Categorization**
```python
def categorize_code(code, metadata):
    categories = []
    
    if 'register_post_type' in code:
        categories.append('custom_post_type')
    if 'add_meta_box' in code:
        categories.append('meta_box')
    if 'register_rest_route' in code:
        categories.append('rest_api')
    if 'register_block_type' in code:
        categories.append('gutenberg')
    # ... more patterns
    
    # Complexity scoring
    complexity = calculate_complexity(code)
    
    # Feature detection
    features = []
    if 'add_action' in code or 'add_filter' in code:
        features.append('hooks')
    if '__(' in code or '_e(' in code:
        features.append('i18n')
    if 'wp_nonce' in code:
        features.append('security')
    
    return {
        'categories': categories,
        'complexity': complexity,
        'features': features
    }
```

### 3.3 Synthetic Generation Pipeline

**Task Generation (Stage 1)**
```python
# Use GPT-4/Claude to generate diverse tasks
system_prompt = """Generate WordPress development tasks with these constraints:
- Must be realistic plugin/theme requirements
- Specify security and accessibility needs
- Include edge cases and constraints
- Vary complexity levels

Format: {
  "task_type": "meta_box|settings_page|...",
  "description": "Natural language requirement",
  "constraints": ["must use nonces", "admin only", "ajax enabled"],
  "complexity": "simple|medium|complex"
}
"""

# Generate 100 tasks per category
tasks = generate_with_llm(system_prompt, category="meta_box", count=100)
```

**Code Generation (Stage 2-3)**
```python
def generate_and_validate(task):
    # Generation
    gen_prompt = f"<wp_gen> {task['description']}\n\nConstraints: {task['constraints']}"
    candidates = []
    
    for model in [claude, gpt4, qwen3_local]:  # via LiteLLM
        code = model.generate(gen_prompt, system=WP_SYSTEM_PROMPT)
        candidates.append(code)
    
    # Validation
    validated = []
    for code in candidates:
        phpcs_result = judge_code(code)
        security_result = check_security(code)
        
        if phpcs_result['passed'] and all(security_result.values()):
            validated.append({
                'task': task,
                'code': code,
                'validation': {
                    'phpcs': phpcs_result,
                    'security': security_result
                }
            })
    
    return validated
```

**Judge Dataset Generation (Stage 4)**
```python
def create_judge_sample(code, is_positive=True):
    # Get automated scores
    phpcs = judge_code(code)
    security = check_security(code)
    i18n = check_i18n(code)
    docs = check_documentation(code)
    
    # Calculate composite scores
    wpcs_score = 100 - (phpcs['errors'] * 10 + phpcs['warnings'] * 2)
    security_score = sum(security.values()) / len(security) * 100
    i18n_score = calculate_i18n_score(i18n)
    docs_score = calculate_docs_score(docs)
    
    overall = (wpcs_score + security_score + i18n_score + docs_score) / 4
    
    # Extract issues
    must_fix = []
    if phpcs['errors'] > 0:
        must_fix.extend(phpcs['violations'])
    if not security['has_nonce_verification']:
        must_fix.append("Missing nonce verification")
    
    return {
        'instruction': f"<wp_judge> Evaluate this WordPress code:\n\n{code}",
        'response': json.dumps({
            'overall_score': overall,
            'wpcs_compliance': wpcs_score,
            'security_score': security_score,
            'i18n_score': i18n_score,
            'documentation_score': docs_score,
            'must_fix_issues': must_fix,
            'passes_threshold': overall >= 80
        }, indent=2)
    }
```

### 3.4 Quality Control

**Automated Checks**
- All code must parse without syntax errors (`php -l`)
- PHPCS must run without crashes
- Security checks must complete
- No duplicate samples (use hash-based deduplication)

**Manual Review Sampling**
- Review 20% of synthetic data
- Check for unrealistic patterns
- Verify explanations are accurate
- Ensure diversity across categories

**Validation Split**
- 80% training
- 10% validation (for monitoring during training)
- 10% held-out test set (for final evaluation)

---

## 4. Training Strategy

### 4.1 MoE Architecture Configuration

**Base Transformer**
- Start from Qwen3-30B-A3B (`Qwen/Qwen3-30B-A3B`) — already a native MoE
- Attention layers: all shared
- FFN layers: already sparse MoE (128 experts, top-8 routing per token)

**MoE Layer Configuration**
```python
moe_config = {
    'num_experts': 8,
    'num_experts_per_tok': 2,  # top-2 routing
    'router_type': 'learned',   # trainable gating network
    'expert_capacity_factor': 1.25,
    'load_balancing_loss_coef': 0.01,
    'router_z_loss_coef': 0.001
}
```

**Task Token Setup**
- Expand vocabulary with 2 special tokens
- Initialize embeddings as average of existing tokens
- Embeddings will specialize during fine-tuning

### 4.2 Training Phases

**Phase 1: Multi-Task Supervised Fine-Tuning**

**Dataset Mixing**
- 50% generation tasks (`<wp_gen>` samples)
- 50% judging tasks (`<wp_judge>` samples)
- Shuffle thoroughly

**Training Config**
```python
training_args = {
    'batch_size': 4,  # per device
    'gradient_accumulation_steps': 16,  # effective batch 64
    'learning_rate': 2e-5,
    'lr_scheduler': 'cosine',
    'warmup_steps': 500,
    'max_steps': 10000,
    'bf16': True,
    'gradient_checkpointing': True,
    'max_seq_length': 2048,
}

lora_config = {
    'r': 64,
    'lora_alpha': 128,
    'target_modules': ['q_proj', 'v_proj', 'gate', 'experts'],
    'lora_dropout': 0.05,
}
```

**Loss Function**
- Standard cross-entropy on next-token prediction
- Add MoE auxiliary losses:
  - Load balancing loss (encourage even expert usage)
  - Router z-loss (prevent routing collapse)

**Phase 2: DPO/RLHF (Optional Refinement)**

If you have preference data (human rankings of outputs):
- Generate N completions for each task
- Have annotators rank them
- Use DPO (Direct Preference Optimization) to refine

### 4.3 Training Infrastructure

**Hardware: DGX Spark (via DGX Toolbox)**
- NVIDIA Blackwell GB10 GPU, 6,144 CUDA cores
- 128GB unified memory (model + data in single address space)
- Ideal for Qwen3-30B-A3B LoRA fine-tuning (~60GB BF16, fits in 128GB unified memory)

**Software Stack (DGX Toolbox Components)**
- **Unsloth Studio** (:8000) — Interactive LoRA/QLoRA fine-tuning with session persistence
- **eval-toolbox** container — lm-eval benchmarks, W&B experiment tracking, torchmetrics
- **data-toolbox** container — polars, duckdb, datatrove deduplication, cleanlab quality filtering
- **vLLM** (:8020) — Batch inference for DPO candidate generation
- **LiteLLM** (:4000) — Unified API for cross-model comparison during eval
- **Safety harness** (:5000) — Guardrails validation, red-teaming, PII redaction

**Training Time Estimates**
- Phase 1 (10K steps, DGX Spark): ~24-36 hours
- Total data: ~15K samples × 3 epochs = 45K examples
- Iterations: 45K / 64 (batch) ≈ 700 steps per epoch

### 4.4 Evaluation During Training

**Automated Metrics (per 500 steps)**

Generator metrics:
- PHPCS pass rate on held-out tasks
- Security check pass rate
- Functional correctness (wp-playground tests for subset)

Judge metrics:
- Correlation with ground-truth scores (Spearman's ρ)
- Classification metrics (precision/recall for pass/fail)
- Mean absolute error on score predictions

**Qualitative Review (per 2000 steps)**
- Manually inspect 20 generation samples
- Check 20 judge outputs for alignment with human judgment
- Look for failure modes or biases

---

## 5. Post-Training Validation

### 5.1 Comprehensive Test Suite

**Generator Evaluation**

Create 500 held-out tasks across all categories:
```python
test_suite = {
    'custom_post_types': 50,
    'settings_pages': 50,
    'shortcodes': 40,
    'rest_api': 40,
    'meta_boxes': 40,
    'gutenberg_blocks': 50,
    'woocommerce': 40,
    'multisite': 30,
    'admin_ajax': 40,
    'cron_jobs': 30,
    'security_focused': 50,  # Tasks explicitly requiring security
    'i18n_focused': 40,      # Tasks requiring translation
}
```

**Metrics:**
- PHPCS pass rate (target: >95%)
- Security checklist pass rate (target: >98%)
- Functional correctness (manual or automated testing)
- Code quality (PHPStan, PHPMD scores)

**Judge Evaluation**

Use 500 held-out (code, scores) pairs:
```python
judge_metrics = {
    'spearman_correlation': target > 0.85,
    'kendall_tau': target > 0.75,
    'classification_precision': target > 0.90,
    'classification_recall': target > 0.88,
    'mae_overall_score': target < 8,
}
```

**Cross-Model Validation**
- Compare judge scores with GPT-4 scores (via LiteLLM)
- Measure agreement on critical issues
- Check for false negatives (missing security issues)

### 5.2 Integration Testing

**End-to-End Workflow**
1. Generate code for complex task
2. Judge evaluates it
3. If score < threshold, iterate
4. Validate final code in wp-playground

**WordPress Playground Integration**
```python
def test_in_playground(code, test_type='plugin'):
    # Start WordPress instance
    playground = start_wp_playground()
    
    # Install code
    if test_type == 'plugin':
        playground.install_plugin(code)
    
    # Run basic checks
    results = {
        'activates': playground.activate(),
        'no_fatal_errors': playground.check_logs(),
        'admin_accessible': playground.test_admin_page(),
    }
    
    playground.cleanup()
    return results
```

### 5.3 Adversarial Testing

**Security Challenge Set**
- Attempt to get model to generate code with known vulnerabilities
- Check if judge correctly identifies them

**Edge Cases**
- Very complex multisite scenarios
- Legacy WordPress compatibility
- Unusual plugin interactions

---

## 6. Model Packaging & Distribution

### 6.1 Repository Structure
````

wp-qwen3-moe/\
├── README.md # Overview, usage, citation\
├── MODEL\_CARD.md # Detailed model card\
├── LICENSE # Apache 2.0 or MIT\
├── config.json # Model configuration\
├── tokenizer.json # Extended tokenizer\
├── tokenizer\_config.json # Tokenizer metadata\
├── special\_tokens\_map.json # Task token definitions\
├── pytorch\_model.bin # Model weights (or safetensors)\
├── training\_args.json # Training hyperparameters used\
├── examples/\
│ ├── generate\_plugin.py # Generation example\
│ ├── judge\_code.py # Judging example\
│ └── end\_to\_end.py # Combined workflow\
├── evaluation/\
│ ├── test\_suite/ # Held-out test tasks\
│ ├── benchmark\_results.json # Evaluation metrics\
│ └── comparison\_with\_baselines.md\
├── data\_card.md # Data sources, collection, ethics\
└── requirements.txt # Dependencies

````
text

### 6.2 HuggingFace Model Card

**Metadata Block**
```yaml
***
language:
- en
license: apache-2.0
library_name: transformers
tags:
- code
- wordpress
- php
- mixture-of-experts
- code-generation
- code-review
base_model: Qwen/Qwen3-30B-A3B
model_type: moe
datasets:
- wordpress/wordpress-core
- custom-synthetic-wp-data
metrics:
- accuracy
- pass@k
model-index:
- name: wp-qwen3-moe-30b
  results:
  - task:
      type: code-generation
      name: WordPress Code Generation
    dataset:
      name: WP-CodeBench
      type: custom
    metrics:
    - type: phpcs-pass-rate
      value: 96.2
    - type: security-pass-rate
      value: 98.5
***
```

**Usage Example**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "your-org/wp-qwen3-moe-30b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Generation
prompt = "<wp_gen> Create a settings page for a newsletter plugin with email validation"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
code = tokenizer.decode(outputs, skip_special_tokens=False)

# Judging
code_to_review = "<?php\nfunction my_form() {\n  echo $_POST['data'];\n}\n"
judge_prompt = f"<wp_judge> Evaluate this WordPress code:\n\n{code_to_review}"
inputs = tokenizer(judge_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
critique = tokenizer.decode(outputs, skip_special_tokens=False)
```

### 6.3 Deployment Formats

**Standard HuggingFace**
- Native Transformers format
- Compatible with `AutoModel` loading
- Works with inference endpoints

**DGX Toolbox Serving Stack (recommended)**
- **vLLM** (:8020) — High-throughput OpenAI-compatible inference, MoE-aware
- **Ollama** (:11434) — Local serving with GGUF quantized models
- **LiteLLM** (:4000) — Unified API proxy routing to vLLM/Ollama/cloud models
- **Open-WebUI** (:12000) — Full-featured chat UI for interactive demo/testing
- **Triton/TensorRT-LLM** (:8010) — Production-grade optimized inference
- **Safety harness** (:5000) — Guardrails, rate limiting, PII redaction gateway

**Deployment via DGX Toolbox**
```bash
# Serve via vLLM (high throughput)
~/dgx-toolbox/inference/start-vllm.sh  # auto-loads from ~/.vllm-model

# Serve via Ollama (GGUF quantized)
ollama run wp-qwen3-moe-30b

# Route through LiteLLM for unified API
~/dgx-toolbox/inference/start-litellm.sh  # config in ~/.litellm/config.yaml

# Interactive demo via Open-WebUI
~/dgx-toolbox/containers/start-open-webui.sh
```

---

## 7. Additional Considerations

### 7.1 Continuous Improvement

**Data Flywheel**
- Collect user-generated code (with permission)
- Run through judge model
- High-scoring code added to corpus
- Low-scoring code used for contrastive learning

**Feedback Loop**
- Track which generations users accept/reject
- Use for preference learning (DPO)
- Refine judge based on human disagreements

### 7.2 Version Management

**Model Versioning**
- v1.0: Initial release (current spec)
- v1.1: Add more WooCommerce patterns
- v2.0: Expand to JavaScript/Gutenberg block code
- v3.0: Multi-lingual support (non-English comments)

**Dataset Versioning**
- Tag dataset with version
- Document changes between versions
- Maintain reproducibility

### 7.3 Community Engagement

**Documentation**
- Comprehensive README
- Video tutorials
- Blog post with benchmarks
- Research paper (optional)

**Demo Applications**
- WordPress plugin generator web UI
- Code review GitHub Action
- VSCode extension

**Feedback Channels**
- GitHub issues for bugs
- Discussions for feature requests
- Discord/Slack community

### 7.4 Legal & Compliance

**Licensing**
- Model: Apache 2.0 (compatible with Qwen3 license)
- Training data: Document all sources
- WordPress code: GPL-compatible

**Attribution**
- Credit WordPress Foundation
- Acknowledge data sources
- List contributors

---

## 8. Timeline & Milestones

### Phase 1: Data Collection (Weeks 1-3)
- Week 1: Extract WordPress Core corpus
- Week 2: Generate contrastive pairs
- Week 3: Synthetic task generation

### Phase 2: Data Preparation (Weeks 4-5)
- Week 4: PHPCS validation, categorization
- Week 5: Judge dataset creation, quality control

### Phase 3: Model Setup (Week 6)
- Convert base model to MoE or select MoE base
- Add task tokens to tokenizer
- Set up training infrastructure

### Phase 4: Training (Weeks 7-8)
- Week 7: Multi-task SFT
- Week 8: Evaluation and refinement

### Phase 5: Validation & Packaging (Weeks 9-10)
- Week 9: Comprehensive testing
- Week 10: Documentation, HF upload

**Total Duration:** 10 weeks

---

## 9. Success Criteria

**Quantitative**
- Generator PHPCS pass rate: >95%
- Generator security pass rate: >98%
- Judge correlation with human scores: >0.85
- Model size: <10B parameters
- Inference latency: <2s per generation (on DGX Spark via vLLM)

**Qualitative**
- Generates idiomatic WordPress code
- Catches critical security issues reliably
- Explanations from judge are actionable
- Community adoption (100+ HF downloads in first month)

**Impact**
- Reduces WordPress development time
- Improves code quality in ecosystem
- Educational value for learners
- Research contribution to code-specific MoE

---

## 10. Risk Mitigation

**Technical Risks**
- MoE training instability → Use established recipes, monitor loss
- Catastrophic forgetting → Maintain base capability validation
- Overfitting to synthetic data → Regular validation on real code

**Quality Risks**
- Model generates insecure code → Security-focused testing, red team
- Judge misses critical issues → Adversarial testing, false negative analysis
- Code doesn't run in real WP → Integration testing in wp-playground

**Adoption Risks**
- Model too large for users → Provide quantized versions
- Hard to use → Comprehensive examples, simple API
- License concerns → Clear licensing, GPL-compatible

---

## Appendix A: Tool Stack Reference

**Data Collection & Processing**
- `git` - Repository cloning
- `PHP_CodeSniffer` - WPCS validation
- `phpstan`, `phpmd` - Static analysis
- Python scripts - Extraction, parsing

**Training (DGX Toolbox)**
- Unsloth Studio (:8000) — Interactive LoRA/QLoRA fine-tuning
- eval-toolbox container — lm-eval, W&B, torchmetrics
- data-toolbox container — polars, duckdb, datatrove, cleanlab, distilabel

**Evaluation (DGX Toolbox)**
- eval-toolbox — lm-eval benchmarks, ragas, W&B dashboards
- vLLM (:8020) — Batch inference for eval sets
- Safety harness (:5000) — Red-teaming, adversarial testing
- wp-playground — Runtime testing

**Distribution (DGX Toolbox)**
- HuggingFace Hub — Model hosting
- Ollama — GGUF quantized local serving
- vLLM / Triton TensorRT-LLM — Production inference
- LiteLLM — Unified API routing
- Open-WebUI — Interactive demo

---

## Appendix B: Dataset Schema Examples

**Generator Training Sample**
```json
{
  "instruction": "<wp_gen> Create a meta box that stores a product SKU with validation",
  "input": "",
  "output": "<?php\n/**\n * Add product SKU meta box\n */\nfunction wpgen_add_sku_meta_box() {\n    add_meta_box(\n        'wpgen_sku',\n        __( 'Product SKU', 'textdomain' ),\n        'wpgen_sku_callback',\n        'product',\n        'side'\n    );\n}\nadd_action( 'add_meta_boxes', 'wpgen_add_sku_meta_box' );\n\n/**\n * Meta box callback\n */\nfunction wpgen_sku_callback( $post ) {\n    wp_nonce_field( 'wpgen_save_sku', 'wpgen_sku_nonce' );\n    $value = get_post_meta( $post->ID, '_wpgen_sku', true );\n    ?>\n    <label for=\"wpgen_sku_field\"><?php esc_html_e( 'SKU:', 'textdomain' ); ?></label>\n    <input type=\"text\" id=\"wpgen_sku_field\" name=\"wpgen_sku_field\" value=\"<?php echo esc_attr( $value ); ?>\" pattern=\"[A-Z0-9-]+\" />\n    <?php\n}\n\n/**\n * Save meta box data\n */\nfunction wpgen_save_sku( $post_id ) {\n    if ( ! isset( $_POST['wpgen_sku_nonce'] ) ) {\n        return;\n    }\n    if ( ! wp_verify_nonce( $_POST['wpgen_sku_nonce'], 'wpgen_save_sku' ) ) {\n        return;\n    }\n    if ( defined( 'DOING_AUTOSAVE' ) && DOING_AUTOSAVE ) {\n        return;\n    }\n    if ( ! current_user_can( 'edit_post', $post_id ) ) {\n        return;\n    }\n    if ( isset( $_POST['wpgen_sku_field'] ) ) {\n        $sku = sanitize_text_field( wp_unslash( $_POST['wpgen_sku_field'] ) );\n        update_post_meta( $post_id, '_wpgen_sku', $sku );\n    }\n}\nadd_action( 'save_post', 'wpgen_save_sku' );\n",
  "metadata": {
    "category": "meta_box",
    "complexity": "medium",
    "features": ["nonce", "capability_check", "sanitization", "i18n"]
  }
}
```

**Judge Training Sample**
```json
{
  "instruction": "<wp_judge> Evaluate this WordPress code:\n\n<?php\nfunction display_user_comment() {\n    echo $_POST['comment'];\n}\n",
  "input": "",
  "output": "{\n  \"overall_score\": 15,\n  \"wpcs_compliance\": 40,\n  \"security_score\": 0,\n  \"performance_score\": 60,\n  \"i18n_score\": 100,\n  \"accessibility_score\": 50,\n  \"documentation_score\": 20,\n  \"must_fix_issues\": [\n    \"CRITICAL: Direct output of $_POST data enables XSS attacks\",\n    \"CRITICAL: No nonce verification for form submission\",\n    \"CRITICAL: No capability check before output\",\n    \"HIGH: Missing input sanitization\",\n    \"HIGH: Missing output escaping\",\n    \"MEDIUM: No PHPDoc comment\",\n    \"LOW: Non-prefixed function name may conflict\"\n  ],\n  \"suggested_improvements\": [\n    \"Verify nonce before processing POST data\",\n    \"Check user capabilities with current_user_can()\",\n    \"Sanitize input with sanitize_textarea_field()\",\n    \"Escape output with esc_html()\",\n    \"Add function prefix to prevent conflicts\",\n    \"Add PHPDoc block describing purpose and parameters\"\n  ],\n  \"passes_threshold\": false,\n  \"explanation\": \"This code has critical security vulnerabilities. It directly outputs unsanitized POST data without nonce verification or capability checks, making it vulnerable to XSS attacks and CSRF. Never trust user input.\"\n}\n"
}
```

---

## Appendix C: PHPCS WordPress Standards Reference

**Standard Levels**
- `WordPress-Core` - WordPress core coding standards
- `WordPress-Extra` - Extended standards (recommended)
- `WordPress-Docs` - Documentation standards
- `WordPress` - All of the above

**Key Rules Enforced**
- Naming: Lowercase with underscores, prefixes
- Spacing: Exact whitespace requirements
- Braces: Required for all control structures
- Yoda conditions: Constants on left
- Nonce verification: Required for forms
- Capability checks: Required for privileged operations
- Sanitization: Input must be sanitized
- Escaping: Output must be escaped
- Internationalization: Use translation functions
- Database queries: Use $wpdb methods, prepare statements

**Installation**
```bash
composer require --dev squizlabs/php_codesniffer
composer require --dev wp-coding-standards/wpcs
phpcs --config-set installed_paths vendor/wp-coding-standards/wpcs
```

---

## Document Version
- Version: 1.0
- Date: March 26, 2026
- Author: Project Specification (for Claude AI)
- Status: Ready for Implementation