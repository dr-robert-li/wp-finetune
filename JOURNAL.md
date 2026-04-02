# Engineering Journal — wp-qwen3-moe

Decisions, reasoning, and observations logged as the project evolves.

---

## 2026-04-03 — run-evaluation skill: autonomous eval with HITL only at the decision point

### The design question

The eval pipeline has multiple stages (E_eff profiling, adapter serving, static eval, wp-bench, triage) totalling ~2 hours of wall-clock time. The original approach required human involvement at several intermediate checkpoints. But most of these checkpoints are binary — the data either passes a hard gate or doesn't. The only genuinely subjective decision is *which surviving ratios to carry forward*, because that requires weighing E_eff (compression potential) against eval quality (task performance) against training loss curves — a multi-dimensional tradeoff where reasonable people could disagree.

### The HITL placement principle

Human-in-the-loop should sit at the **highest-leverage decision point**, not at every stage. For eval triage:

- **Gate 1 (E_eff trend):** Numeric — is the trend downward as gen fraction increases? Binary, automatable.
- **Gate 2 (hard thresholds):** PHPCS >95%, Spearman >0.85, Security >98%, plus 5pp rule — binary elimination, automatable.
- **Human review (Gate 3):** Full comparison table with all signals — the human picks survivors. This is the gate that matters, because it encodes judgment about which quality-vs-compression tradeoff to take into Phase 7.

Everything before Gate 3 is autonomous. Everything after Gate 3 (state update, next-steps routing) is autonomous. The human touches the pipeline exactly once, at the point where their judgment actually changes the outcome.

### Skill architecture

The skill follows the `run-training` pattern (6-step lifecycle, DGX Toolbox container management, idempotent completion markers):

```
Step 0:  Inventory + DGX validation                        [autonomous]
Step 1:  Base-model E_eff profiling (5 ratio distributions) [autonomous]
         → Decision Gate 1: trend check, gate 60/40 training
Step 2:  Sequential adapter eval (vLLM serve → eval suite)  [autonomous]
Step 3:  Automated triage (GATE-02 elimination)             [autonomous]
Step 4:  ► HUMAN REVIEW — full comparison table             [STOP]
         → Human picks survivors for Phase 7
Step 5-6: State update + next steps routing                 [autonomous]
```

Each step writes a `.complete` marker. Re-running the skill resumes from the last incomplete step. `--force` bypasses all markers for a full re-run. This idempotency was critical — the first run hit a router hook bug (see below) and the retry needed to skip the inventory/validation steps that had already passed.

### The router hook bug

The profiling script hooks `model.layers[i].mlp.gate` — a `nn.Linear` that outputs raw router logits of shape `[n_tokens, n_experts]`. The original hook expected the output to be a tuple `(logits, scores, indices)` matching the parent `MoeBlock.forward()` signature. But `register_forward_hook` on the gate Linear receives only the Linear's own output (a single tensor), not the MoeBlock's post-processing.

Fix: compute `top_k` from the raw logits inside the hook via `torch.topk(outputs, k=self.top_k, dim=-1).indices`. The hook now works at the correct abstraction level — it observes gate decisions, not post-gate routing.

### Container dependency resolution

The DGX Spark environment has a persistent friction point: the host Python has CPU-only torch (CUDA=False), so all GPU work must run inside Docker containers. The `unsloth-headless` container image ships without `transformers` pre-installed (the sync script installs deps at startup), and the NGC `pytorch:25.11-py3` base has a `keras_nlp` stub that conflicts with `transformers>=4.56`. Resolution: uninstall the keras stubs, then install the full dependency chain. This is fragile — a proper eval-toolbox Docker image with pre-baked deps would eliminate this class of startup failures.

### Why triage, not winner selection

Phase 4 deliberately does NOT pick a winner. A ratio with slightly lower eval score but sharper routing concentration (lower E_eff) may produce a better production model after MoE-Sieve + pruning in Phase 7-8. The triage preserves these candidates — it only eliminates ratios that are clearly non-viable (hard gate failures, >5pp behind). Phase 7 profiles the fine-tuned adapters (not the base model) and makes the final selection using both eval quality AND adapter E_eff, which captures how LoRA actually shifted routing.

---

## 2026-04-03 — Restructuring the evaluation-compression-packaging pipeline

### The problem

The original roadmap had Phase 4 picking a single winning gen/judge ratio, then sending that one ratio through MoE-Sieve → GRPO → pruning → packaging. This assumed the best pre-compression model is also the best post-compression model. That assumption is wrong.

Each ratio will have different routing concentration — 30/70 (judge-heavy) likely activates different expert clusters than 50/50 (balanced). A ratio that scores slightly lower on raw eval might have sharper routing and compress to a smaller, faster model with equivalent quality after pruning. The metric that matters for production is **post-compression quality-per-VRAM**, not pre-compression quality.

But carrying all ratios through the full pipeline is a combinatorial explosion: 3 ratios × 3 k-budgets × 3+ seeds = 27+ MoE-Sieve runs, weeks of GPU time for what amounts to hyperparameter search on a single variable.

### The funnel architecture

The solution is a two-stage funnel that collapses the search space at the cheapest possible decision points:

```
Phase 4 (Triage)          Phase 7 (Profile + Select)       Phase 8+ (Single-track)
─────────────────         ──────────────────────────       ──────────────────────
30/70 ──eval──┐           ┌──profile──E_eff=18──┐
40/60 ──eval──┼─survive?──┼──profile──E_eff=14──┼─lowest E_eff──→ winner → MoE-Sieve → GRPO → prune
50/50 ──eval──┘           └──profile──E_eff=22──┘  at ≡ quality
```

- **Phase 4 (triage):** ~hours of eval inference (cheap, parallelisable). High bar for elimination — only cut ratios that fail hard gates or are >5pp behind the best. A 1-2pp difference may invert after pruning if routing concentration differs.
- **Phase 7 (profile all survivors):** ~minutes of gradient-free forward passes per ratio (essentially free). Compute E_eff per layer per ratio.
- **Phase 7→8 gate:** Decision matrix combining eval score and E_eff. Lowest E_eff at equivalent quality wins.
- **Phase 8+:** Single-track from there.

### Routing concentration and effective expert count — the math

For a given MoE layer `l` with `E` experts, the **routing entropy** measures how uniformly tokens are distributed across experts:

```
H_l = -Σ_{e=1}^{E} p_{l,e} · log(p_{l,e})
```

where `p_{l,e}` is the fraction of tokens routed to expert `e` in layer `l`. Lower entropy = sharper routing = more experts can be pruned safely. Bounds: `H = 0` (all tokens to one expert) to `H_max = log(E) ≈ 4.85` for `E = 128` (uniform distribution).

But raw entropy doesn't capture tail shape — two distributions with identical entropy can have very different pruning profiles (10 hot + 118 dead vs 40 warm + 88 cool). The first prunes to 10 experts trivially; the second has no clean cutoff.

The better scalar is **effective expert count** — the exponential of entropy:

```
E_eff_l = exp(H_l) = exp(-Σ_{e=1}^{E} p_{l,e} · log(p_{l,e}))
```

This directly gives the "effective number of experts" per layer. If `E_eff = 15` across layers, ~15 experts carry the load and the other 113 are pruning candidates. If `E_eff = 80`, routing is diffuse and aggressive pruning will hurt quality.

For the REAP saliency scoring comparison, the per-expert importance is:

```
S_j = mean_{x ∈ active(j)}( g_j(x) · ‖f_j(x)‖₂ )
```

where `g_j(x)` is the gating score and `f_j(x)` is the expert output — computed only over tokens where expert `j` is actually active. REAP captures both frequency and impact; E_eff captures only frequency. The two signals are complementary — E_eff predicts pruning headroom, REAP scores which specific experts to remove.

For AIMER, the weight-only scoring per expert is:

```
AIMER(expert) = P / √(N · Q)
```

where `P = Σ|w_i|` (L₁ norm), `N` = parameter count, `Q = Σ w_i²` (squared Frobenius norm) across gate/up/down projection matrices. Scale-invariant, bounded `[1/√N, 1]`, related to Hoyer sparsity.

The decision matrix per surviving ratio:
- **Eval score** (from Phase 4, normalised 0-1)
- **Mean E_eff** across all MoE layers — overall routing concentration
- **Max E_eff** across layers — worst-case bottleneck, constrains pruning ceiling
- **E_eff variance** across layers — predicts whether uniform pruning ratio per layer works or layer-adaptive pruning is needed (low variance = uniform is fine)

### Edge case: judge-activating general-knowledge experts

If the best-quality ratio has significantly diffuse routing (E_eff > 30), that signals judge data is activating general-knowledge experts — experts that carry reasoning capabilities from pretraining that help evaluate code quality. Those experts are genuinely load-bearing for judge but irrelevant for gen. Since GRPO only refines gen (judge frozen from SFT), and the production model's judge quality is already fixed before pruning, this favours gen-weighted ratios that concentrate routing and compress cleanly.

### Phase 12 now runs AIMER vs REAP as a sub-experiment

Both pruning methods run on the same merged post-GRPO model:
- **AIMER** (~1 sec per ratio, no calibration) at 25/50/75% — task-agnostic baseline
- **REAP** (~3 hrs, WordPress calibration data) at 25/50/75% — domain-aware comparison
- 6 variants total evaluated via gating mask across all 9 dimensions
- Domain specificity analysis: expert overlap between methods quantified per layer — the key question is whether WordPress routing concentration is distinct enough from general code for REAP to differentiate

The answer feeds directly into the model card (PKG-04) as a published finding.

### Base-model profiling before eval — a further reordering

A second insight emerged: we already have 3 ratios fully SFT'd (30/70, 40/60, 50/50) but not 60/40 or 70/30. Base-model profiling is cheaper than eval (~minutes vs hours) and answers a question that changes what Phase 4 even needs to do.

If we profile the base model with all 5 ratio data distributions first, we can check whether E_eff trends downward as the gen fraction increases. If it does, that's a signal that gen-heavier ratios concentrate routing more sharply (fewer experts needed → better compression) and it's worth training 60/40 and 70/30 despite the ~60+ hours of GPU time each. If E_eff is flat or trending up, those ratios won't compress better and aren't worth the investment.

The reordered execution flow:

```
Phase 4 Step 1: Base-model profile all 5 ratio data distributions (~minutes)
         │
         ├─ E_eff trending down at 60/40, 70/30? → Start training in background (~2 days)
         │
Phase 4 Step 2-3: Eval existing 30/70, 40/60, 50/50 adapters (in parallel with any training)
         │
Phase 4 Step 4: Triage — eliminate clear failures, carry survivors
         │
Phase 7: Profile fine-tuned ADAPTERS (routing shifted by LoRA — different signal than base-model)
         │
Phase 7→8 gate: Lowest adapter E_eff at equivalent quality → winner
         │
Phase 8+: Single-track MoE-Sieve → GRPO → prune → package
```

The key distinction: Phase 4 does base-model profiling (cheap, gates whether to train more ratios), Phase 7 does fine-tuned adapter profiling (captures how LoRA shifted routing, which is the actual signal for MoE-Sieve targeting). These are complementary — the base-model E_eff predicts compression potential before training, the adapter E_eff measures it after.

### What changed in the roadmap

| Phase | Before | After |
|-------|--------|-------|
| 4 | Pick single winner | Base-model profiling first (gates 60/40, 70/30 training), then eval triage on available adapters |
| 7 | Profile winner only | Profile fine-tuned adapters of ALL survivors, compute adapter E_eff, select via decision matrix |
| 7→8 | (implicit) | Explicit gate: lowest adapter E_eff at equivalent quality (within 2pp) |
| 12 | REAP only | REAP + AIMER side-by-side, 6 variants, domain specificity analysis |

New requirements: PROF-05 (profile all survivors), GATE-01 (decision matrix), GATE-02 (triage thresholds), PRUNE-06 (physical removal + model card).

---

## 2026-04-03 — AIMER: calibration-free pruning as a REAP alternative, and why we'll test both

### The paper

[AIMER: Calibration-Free Task-Agnostic MoE Pruning](https://arxiv.org/abs/2603.18492) (Liu et al., March 2026, Zhejiang/Westlake/MBZUAI) proposes a pruning method that ranks experts purely from pretrained weights — no calibration data, no forward pass, no routing traces. It scores each expert by computing a normalized weight statistic:

```
AIMER(expert) = P / √(N × Q)
```

where P = sum of L₁ norms across gate/up/down projections, N = parameter count, Q = sum of squared Frobenius norms. Experts with the *highest* scores (most distinctive weight distributions) are retained; lowest-scoring experts are pruned. The score is scale-invariant and bounded [1/√N, 1], related to the Hoyer sparsity metric applied at expert granularity.

The key selling points vs REAP:

| Property | AIMER | REAP |
|----------|-------|------|
| Calibration data | None | Yes (4.2M tokens from C4) |
| Scoring time | 0.22–1.27 seconds | 0.75–2.96 hours |
| Peak memory | 40–57 GB | 44–63 GB |
| Input signal | Weight matrices only | Gate values × activation norms |
| Sensitivity to data | None (deterministic) | Varies with calibration sample size |

### Why it caught my attention

AIMER's coding benchmark results on Qwen3-30B-A3B at 50% pruning are striking: code average 36.1% vs REAP's 4.6%. At 25% pruning on ERNIE-21B, it improves code average by 29.7pp over the strongest baseline (45.5% vs 15.8%). These are large margins on the exact model family and task domain we care about.

The calibration sensitivity point is also significant. REAP's published results show that varying C4 sample size from 0.5M to 2.1M tokens produces substantially different pruning outcomes. For our use case, we'd be calibrating on WordPress data — a narrow distribution that may amplify this sensitivity. AIMER sidesteps the question entirely by ignoring activations.

### Why it's not a slam dunk for us

AIMER is explicitly **task-agnostic**. It identifies experts that are "distinctive" in weight space, not experts that are *useful for a specific task*. The paper acknowledges this: "For task-specific compression, calibration sets remain necessary."

Our situation is unusual. After MoE-Sieve SFT + GRPO, the model's hot experts have been fine-tuned specifically for WordPress code. The routing distribution has shifted. The experts that matter most to us are the ones that fire on `<wp_gen>` and `<wp_judge>` prefixed inputs — a routing-driven signal that AIMER cannot see by construction. REAP's saliency scoring (gate_value × activation_magnitude on WordPress calibration data) directly measures what we care about: which experts contribute the most *when processing WordPress code*.

The tension is:
- **AIMER** may be better at preserving general expert quality (avoiding calibration-induced bias), but it's blind to our fine-tuned routing distribution
- **REAP** directly measures task-relevant expert importance, but is sensitive to the calibration set and slower to compute

For a domain-specialized model where we've deliberately concentrated routing onto a subset of experts, REAP's task-aware scoring *should* be more precise. But AIMER's massive coding benchmark advantage suggests its weight-based signal captures something REAP's activation traces miss — possibly structural redundancy patterns in the weight matrices themselves.

### Decision: experiment with both

REAP and AIMER are both cheap to run (REAP: a single calibration forward pass; AIMER: seconds of weight arithmetic). Neither involves training. Running both on the merged post-GRPO model and comparing pruning masks adds negligible time to the pipeline.

The roadmap now includes both in Phase 12:
- Run REAP with WordPress calibration data (as originally planned)
- Run AIMER on the same merged model (weight-only, no calibration)
- Compare pruning masks: Jaccard overlap, which experts each method retains/removes
- Evaluate both pruned models on wp-bench and all 9 dimensions at 25%/50%/75% compression
- Pick the winner empirically — or discover that a hybrid (AIMER for cold-expert identification + REAP for borderline experts) works best

If the pruning masks diverge significantly, that itself is interesting data: it means weight-space distinctiveness and routing-space importance are measuring different things, and the right pruning criterion for domain-specialized MoE models is an open question.

### Caveats from the paper

- No formal theoretical justification for why the normalized scoring works — the authors speculate about preserving layer-wise signal statistics but acknowledge this needs rigorous analysis
- Evaluation limited to models ≤30B params
- Minor regressions on creative writing (-0.3%) and multiple-choice QA (-0.6%) relative to best baselines — not relevant to our use case but worth noting
- The paper prunes *pretrained* models, not fine-tuned ones — our post-SFT/GRPO weight distributions may behave differently under AIMER scoring

---

## 2026-04-02 — v2.0 MoE-Sieve & Expert Pruning: why selective training and structural pruning matter

### The goal

After v1.0's full-LoRA fine-tune and v1.1's adaptive training infrastructure, the next question is: can we produce a WordPress-specialized model that is not only accurate but *small and fast enough to self-host on lightweight infrastructure*? A 30B MoE model is powerful but expensive to serve. If most of its 128 routed experts per layer are irrelevant to WordPress/PHP tasks, we're paying inference cost for dead weight.

The v2.0 milestone chains two complementary techniques — MoE-Sieve (selective expert training) followed by REAP (structural expert pruning) — to produce a model that is maximally specialized for WordPress code generation and review, with a dramatically smaller footprint than the full model. The sequence matters: train only the experts that matter, then physically remove the ones that don't.

### Why this matters for LLM research

This implementation demonstrates several frontier techniques in combination that, to my knowledge, haven't been applied together in a published domain-specific fine-tuning pipeline:

**1. Routing-guided selective LoRA on a production-scale MoE.** MoE-Sieve has been validated on OLMoE-1B-7B (64 experts) and Qwen1.5-MoE-A2.7B (60 experts). Applying it to Qwen3-30B-A3B (128 routed + 4 shared experts per layer) is a scale jump that tests whether the 25% expert budget holds at larger architectures — or whether domain concentration allows an even smaller budget. The k-sweep across 10%, 25%, and 50% budgets will produce the first published data point on MoE-Sieve scaling behavior at 128-expert layers.

**2. Task-token-aware expert profiling.** The MoE-Sieve paper profiles experts per-task and confirms low cross-task overlap (Jaccard = 0.13 between MBPP and Wikitext). We extend this by profiling per *task token* within the same domain — `<wp_gen>` (code generation) vs `<wp_judge>` (code review) — and feeding different training data to each expert set. Gen-hot experts see only golden signal (high-quality code), while judge-hot experts see the full quality spectrum (passed + failed + contrastive pairs). This task-aware data filtering is a novel contribution: it treats the MoE router as a natural task decomposer and tailors the training signal accordingly.

**3. Selective training → structural pruning pipeline.** MoE-Sieve identifies which experts are hot (worth training) and which are cold (frozen). REAP then scores those cold experts for structural removal. The combination is more principled than either technique alone: MoE-Sieve's routing profile provides a strong prior on which experts REAP should consider pruning, and REAP's saliency scoring (gate_value × output_norm) provides a more rigorous removal criterion than routing count alone.

**4. Domain-specific MoE compression.** General-purpose MoE pruning (what REAP's published benchmarks test) must preserve broad capability across math, code, language, reasoning. Domain-specific pruning can be far more aggressive because the model only needs to excel at WordPress/PHP. This is the "specialist vs generalist" advantage: a WordPress expert doesn't need the experts that handle Japanese grammar or organic chemistry. The routing concentration for PHP/WordPress data should be sharper than any general benchmark, enabling higher compression ratios without quality loss.

**5. Self-hostable specialist model.** The end state — a pruned, specialized MoE exported as GGUF/AWQ — demonstrates that frontier MoE architectures can be compressed into models runnable on modest hardware (single consumer GPU or small cloud instance) when the domain is narrow enough. This challenges the assumption that MoE models are inherently expensive to serve.

### Research papers and how they shaped the design

**MoE-Sieve: Routing-Guided Selective LoRA** (arXiv 2603.24044, March 2025)

The foundational paper for this milestone. MoE-Sieve's core insight is that in Mixture-of-Experts models, expert activation is highly skewed — a small fraction of routed experts handle the vast majority of tokens for any given task. Training all experts equally via full LoRA wastes compute on cold experts that receive sparse, inconsistent gradient updates. These updates act as noise rather than signal, actually *destabilizing* training (seed-to-seed variance drops 41-64% when cold experts are excluded).

Key findings that directly informed our requirements:

- **25% budget sufficiency.** On both OLMoE and Qwen1.5-MoE, training only the top 25% most-routed experts per layer matched full-LoRA accuracy within ±1pp across Spider, GSM8K, and HellaSwag (5 of 6 conditions passed formal TOST equivalence testing at ε=2pp). This established our starting hypothesis, though the 25% threshold is validated only on 60-64 expert architectures — hence the k-sweep requirement for our 128-expert model.

- **Count-based ranking preferred for Qwen.** The paper tested both count-based (how often an expert is selected) and mass-based (sum of gating weights) expert ranking. For Qwen architecture, Jaccard similarity between methods was 0.920 — nearly identical, but count-based is simpler and recommended. This determined PROF-01's specification.

- **10% subsample stability.** Profiling on just 10% of training data produced expert rankings with Jaccard ≥ 0.94 vs the full dataset. This is critical for our setup: profiling 128 experts × 28 layers on the full WordPress training set would be expensive. The 10% subsample makes the profiling pass cheap.

- **Per-layer selection, not global.** Routing skew amplifies ~2× from early to deep layers. Selecting experts per-layer (not a global top-k) preserves early-layer diversity while exploiting deep-layer concentration. This shaped PROF-04's layer-depth skew analysis requirement.

- **Cross-task expert overlap is low** (Jaccard = 0.13 between MBPP and Wikitext). This is the empirical basis for profiling `<wp_gen>` and `<wp_judge>` tokens separately. If even unrelated tasks (code vs text) show low overlap, our two task types within the same domain will likely activate different expert subsets — justifying task-specific data filtering.

- **Cold-expert noise hypothesis.** The paper's most counterintuitive finding: excluding cold experts doesn't just save compute — it *improves stability*. Cold experts receive too few gradient updates to learn meaningful patterns, and the sparse updates they do receive introduce variance. This validated our decision to freeze (not just down-weight) cold experts during MoE-Sieve training.

- **Training hyperparameters.** The paper used LoRA r=32, α=64, dropout=0.05, which matches our existing v1.0 config exactly — no hyperparameter changes needed for MoE-Sieve, only the target module selection changes.

**What MoE-Sieve doesn't address:** The paper focuses on *training* efficiency — which experts to adapt. It doesn't remove cold experts from the model. A MoE-Sieve-trained model still has all 128 experts per layer at inference time, paying full routing and memory costs. This is where REAP comes in.

**REAP: Routing-Expert-Aware Pruning** (Cerebras Research, 2025)

REAP addresses the complementary problem: after MoE-Sieve identifies which experts are hot vs cold, REAP provides a principled method to *physically remove* the cold ones, producing a genuinely smaller model.

REAP was selected over EASY-EP (the other MoE pruning method initially considered) for three reasons:

1. **Native Qwen3-30B-A3B support.** REAP's published results include Qwen3-30B-A3B with exact metrics: 95.9% accuracy retention at 50% expert pruning on EvalPlus, and model size reduced from 30.5B to 17.3B parameters. EASY-EP was designed for DeepSeek-R1 and would require architecture adaptation.

2. **Saliency-based scoring.** REAP computes expert importance as S_j = mean(g_j(x) · ‖f_j(x)‖₂) — the average product of the gating score and output norm, computed only over tokens where the expert is actually active. This captures both *how often* an expert is selected (frequency) and *how much it contributes when selected* (impact). A rarely-routed expert that produces high-magnitude outputs when activated scores higher than a frequently-routed expert with negligible outputs. This matters for edge case preservation: a security-related expert might fire rarely but critically.

3. **Pruning > merging for generative tasks.** REAP's key finding is that expert *merging* (combining cold experts into hot ones) hurts generative quality, even though it preserves more parameters. The information in cold expert weights is apparently noise for the tasks where hot experts dominate. This aligns with MoE-Sieve's cold-expert noise hypothesis and supports our approach of clean removal rather than knowledge distillation.

**EASY-EP** (RUCAIBox, 2025) was the initial pruning candidate. It uses a similar gating score × output magnitude formula and a two-step mask-then-prune pipeline. However, its codebase targets DeepSeek-R1/V3 architectures natively, and adapting it to Qwen3's router structure would add engineering work without clear accuracy advantages over REAP's native Qwen3 support. EASY-EP remains a valid alternative if REAP produces unexpected results, but for v2.0 we're proceeding with REAP only to reduce scope.

### The full pipeline: SFT → GRPO → Merge → REAP

The techniques chain in a specific order, and that order matters. The critical insight is that REAP must come *after* GRPO, not before — because GRPO changes which experts matter.

1. **Profile** (MoE-Sieve PROF-01 through PROF-04): Forward pass over WordPress data, hooking `Qwen3MoeSparseMoeBlock` gating outputs. Record per-layer, per-task-token expert activation counts. Output: routing heatmap with gen/judge affinity tags.

2. **SFT with MoE-Sieve** (SIEVE-01 through SIEVE-05): Apply LoRA adapters only to hot experts (attention, routers, and shared experts always included). K-sweep at 10%, 25%, 50% budgets to find the accuracy plateau. Train with task-aware data filtering.

3. **GRPO with hot-experts-only + RSPO stabilisation**: Refine `<wp_gen>` generation against verifiable rewards. LoRA targets remain the same hot expert set — router weights and attention are updated as part of the shared parameters being trained.

4. **Merge LoRA into base weights**: Bake the adapter weights into the hot experts' base weights to produce a unified model. This is required before REAP because REAP measures activation magnitude of expert outputs — if the LoRA is still a separate adapter, activation norms won't reflect the true fine-tuned expert contributions.

5. **REAP pruning on final merged model** (PRUNE-01 through PRUNE-05): Re-profile routing on WordPress calibration data (the routing distribution has shifted during GRPO). Score all experts by gate_value × activation_magnitude on the *post-GRPO, post-merge* model. Prune bottom 50-75% of routed experts. Brief router re-normalisation to adjust softmax for removed expert slots.

6. **Validate** (EVAL-01, EVAL-02): A/B compare against v1.0 full-LoRA on wp-bench. Confirm no regression on any of the 9 evaluation dimensions, especially D2_security.

7. **Package** (PKG-01 through PKG-05): Gated compression — eval pruned bf16 first, quantize only if deployment constraints require it.

### Why REAP must follow GRPO, not precede it

Deferring REAP until after GRPO is the correct ordering for a straightforward reason: GRPO will change which experts matter, and you want to prune based on the final routing distribution, not an intermediate one.

**The routing distribution shifts during GRPO.** Even with hot-experts-only GRPO and RSPO stabilisation, the router weights still receive gradient updates — they're part of the attention/shared parameters being trained. This means the relative importance ranking among hot experts will shift during GRPO. An expert that was the 30th-most-routed during SFT profiling might become the 5th-most-important after GRPO discovers it's particularly useful for generating correct WordPress hook patterns. If you prune before GRPO, you risk removing experts that RL would have promoted.

**REAP's scoring inputs change after GRPO.** REAP scores experts by S_j = mean(g_j(x) · ‖f_j(x)‖₂) on a calibration set. Both quantities change after GRPO:
- Gate values shift because the router is updated during GRPO (it's not in the frozen cold-expert set)
- Activation magnitudes shift because the hot experts' LoRA weights change their output distributions

Pruning on pre-GRPO scores would be like measuring furniture before renovating the room.

**REAP is designed for post-training anyway.** REAP is explicitly a one-shot, post-training method — you run it on a finished model with a small calibration set (25-100 samples) and prune in one pass. There's no training loop, no gradient computation. It takes minutes. There's zero benefit to doing it earlier, and real risk in doing so.

**Don't prune before merge.** The LoRA adapter must be merged into the base weights before running REAP. REAP measures activation magnitude of expert outputs — if the LoRA is still a separate adapter, the activation norms won't reflect the true fine-tuned contributions. Merge first, then profile and prune on the unified model.

### Expected compression

REAP on Qwen3-Coder-480B achieved near-lossless compression at 50% expert pruning specifically on code generation tasks. Our WordPress model should compress even more aggressively than a general code model because the domain is narrower — WordPress is a strict subset of PHP/web development, so the routing concentration will be tighter. 60-75% pruning with minimal degradation is realistic, which would take the 30B total parameter count down to roughly 8-12B while maintaining the same ~3B active parameters per forward pass.

### Key design decisions and their rationale

**Task-aware data filtering (SIEVE-02/03).** This is the most novel requirement. The insight: `<wp_gen>` experts should only see high-quality code because their job is generation — exposure to failed examples teaches patterns we don't want generated. `<wp_judge>` experts must see the full quality spectrum because their job is discrimination — they need calibration on what bad code looks like. The MoE router naturally separates these expert populations (different tokens route differently), so we can tailor the training signal without manual expert assignment.

**K-sweep instead of fixed 25% (SIEVE-04/05).** The MoE-Sieve paper's 25% finding comes from 60-64 expert models. Our 128-expert Qwen3-30B-A3B might plateau earlier (WordPress routing is likely more concentrated than general benchmarks) or later (scale effects). Rather than assume, we test 3 budgets and pick the smallest that matches full-LoRA. This produces a publishable data point on MoE-Sieve scaling.

**REAP over EASY-EP.** Both methods use similar saliency formulas. REAP wins on native Qwen3 support (tested, published results) and the finding that pruning outperforms merging for generative tasks. Engineering pragmatism: use the tool that already works with your architecture.

**Conservative pruning with dimension-level validation.** We don't hardcode a pruning ratio. REAP scores experts, we apply masks at multiple compression levels, and the eval suite (particularly D2_security) determines the final ratio. An expert that fires on 0.1% of tokens but catches SQL injection patterns is worth keeping. The saliency scoring helps here — such an expert would have high output magnitude when active, scoring above purely-dead experts.

### Gated compression roadmap: why we don't blindly triple-compress

There's a temptation to chain every compression technique available: MoE-Sieve (selective training, ±1pp) → REAP pruning (~4% loss at 50%) → quantization (~1-3% loss at Q4). Each is individually justified. But compounding them blindly risks stacking quality losses that individually pass eval gates but collectively degrade the model beyond acceptable thresholds.

The solution is a **cascading gate architecture** where each compression stage earns its place independently, and the decision to proceed to the next stage is made *after* evaluating the current one — not assumed in advance.

**Gate 1: Post-REAP pruned bf16.** After MoE-Sieve SFT, GRPO refinement, LoRA merge, and REAP pruning, evaluate the pruned bf16 model across all 9 dimensions and record model size, inference speed, and VRAM requirements. This is the quality baseline for everything that follows. With 60-75% REAP compression on Qwen3-30B-A3B (realistic given WordPress's narrow domain concentration), the model drops from ~30B to roughly 8-12B total parameters while maintaining ~3B active parameters per forward pass — already a dramatic size reduction. If the pruned bf16 model already fits within target deployment constraints (VRAM budget, throughput requirements, download size), there is no reason to quantize at all. Ship bf16.

**Gate 2: Quantization necessity assessment.** Three factors determine whether quantization is warranted:
- *Ollama distribution:* Users on consumer hardware (16-24 GB VRAM) may need GGUF Q4/Q5 even after pruning — pruned bf16 at ~8-12B params still requires ~16-24 GB in bf16 format, which may or may not fit depending on target hardware.
- *vLLM throughput:* AWQ/FP8 on the serving infrastructure enables higher concurrent request throughput even when the pruned model fits in memory at bf16.
- *HuggingFace download size:* Pruned + quantized is a friendlier download for the community.

If none of these apply (e.g., the pruned model is small enough for target hardware at bf16), skip quantization entirely. The gate exists precisely to prevent unnecessary compression.

**Gate 3: Incremental quantization testing.** If quantization is warranted, test from the lightest compression downward: Q8 → Q6 → Q5 → Q4. Evaluate each level against the Gate 1 bf16 baseline (not against v1.0 full-LoRA — the pruned bf16 model is the relevant reference). Stop at the lowest quantization level that holds within ±2pp of Gate 1 on all 9 dimensions. If Q8 is sufficient, never test Q4. If even Q8 regresses beyond threshold, ship pruned bf16 and let users quantize to their own tolerance.

This gated approach has a research benefit beyond engineering prudence: it produces a **compression lineage** — eval results at each stage (full-LoRA → MoE-Sieve SFT → GRPO → merge → REAP pruned bf16 → quantized) showing exactly where quality degrades and by how much. This data is valuable for the community: anyone applying similar techniques to different domains can use our lineage to estimate their own compression budget.

The model card on HuggingFace will document this full lineage: base model → MoE-Sieve selective training (k experts, which budget, eval results) → GRPO refinement (reward metrics, eval results) → LoRA merge → REAP pruning (compression ratio, eval results) → quantization level (if applied, eval results). Transparency about what was compressed and what it cost.

### What this could demonstrate

If successful, v2.0 produces a WordPress specialist model that:
- Matches full-LoRA accuracy (±1pp on wp-bench)
- Has ~70% fewer trainable parameters during fine-tuning (MoE-Sieve)
- Is refined via GRPO against verifiable WordPress rewards before any structural changes
- Has 60-75% fewer total parameters at inference (REAP pruning on the post-GRPO model, ~30B → 8-12B total, ~3B active unchanged)
- May be further compressed via quantization if deployment constraints require it — but only through an eval-gated process, not by default
- Runs on a single consumer-grade GPU or modest cloud instance
- Preserves edge case handling (security, accessibility) through conservative pruning with dimension-level validation
- Ships with a full compression lineage documenting quality at every stage

This would be, to my knowledge, the first published example of chaining routing-guided selective LoRA, reinforcement learning (GRPO), and structural expert pruning (with optional quantization) for domain-specific MoE compression — demonstrating that frontier MoE models can be narrowed into efficient specialists without sacrificing quality on the target domain. The pipeline ordering (SFT → GRPO → merge → REAP → optional quantization) is itself a methodological contribution: it ensures pruning decisions are made on the final routing distribution, not an intermediate one. The gated compression roadmap provides a template for disciplined multi-stage model compression that other practitioners can adapt.

---

## 2026-04-01 — v1.1 Adaptive Training Infrastructure: from patch to milestone

### Why this became a milestone

When the adaptive planner first appeared in v1.0 (Step 8.5 of run-training, March 29), it was 40 lines of inline logic in a skill file: classify thermal zone, bump batch, done. It worked for exactly one run before Run 2's OOM crash exposed the fundamental design flaw — the planner treated the GPU as a single-axis optimisation problem (temperature), when the DGX Spark actually presents a two-axis problem with radically different risk profiles on each axis.

The OOM investigation (March 31) was the turning point. The telemetry showed the system crashing at 119.6 GB / 119.7 GB total — 0.1 GB headroom — while GPU temperature was a comfortable 65°C. The adaptive planner had seen COOL zone, scaled up aggressively (batch 4→8, workers 4→8), and driven the system straight into the memory cliff. The asymmetry between thermal and memory risk on unified memory wasn't something a patch could address. It needed a proper engineering response: new abstractions, new signals, new safety layers, a config-driven threshold system, cross-project infrastructure, and peer review. That's a milestone, not a hotfix.

Treating it as v1.1 rather than a v1.0 patch was the right call for three reasons. First, the scope is genuinely new capability — power-primary routing, a 5-rung thermal exploitation ladder, batch/grad_accum coupling, Unsloth override detection, warmup probes, failure classification, and anchor-based config history. None of this existed in v1.0's temperature-zone logic. Second, it has an external dependency (dgx-toolbox Phase 13 telemetry package) that belongs in a separate release cycle. Third, treating it as a milestone forced proper requirements definition (13 ADPT/BTCH/TELE/PROB requirements), cross-AI plan review, and phased execution — discipline that a "quick fix" would have skipped.

### Research that shaped the design

The design wasn't invented from first principles. It was synthesised from four distinct research threads:

**Thread 1: The OOM post-mortem (March 31).** The debug investigation (`oom-training-dgx-spark.md`) established the core constraint: on DGX Spark's unified memory, CUDA allocation failures occur in the driver's internal path (NV_ERR_NO_MEMORY), below where PyTorch could catch them. The system freezes — no Python exception, no graceful recovery, no SSH. This is [pytorch/pytorch#174358](https://github.com/pytorch/pytorch/issues/174358), a known open issue. The investigation identified four compounding memory creep mechanisms: PyTorch caching allocator fragmentation (up to 43% waste reported in large-scale LLM training), checkpoint save spikes (serializing optimizer state temporarily pins ~14 GB), DataLoader worker accumulation (each subprocess gradually grows resident memory), and the sawtooth respawn pattern (workers killed and re-spawned allocate fresh buffers each cycle). This research directly produced the `persistent_workers=True` fix, the peak-based headroom calculation, and the requirement that the planner must treat memory as a separate budget from thermal.

**Thread 2: Power vs temperature as the primary signal.** The v1.0 planner used temperature zones (COLD/COOL/WARM/HOT/CRITICAL). Research into the GB10's thermal behaviour showed this was backwards. Temperature is a lagging indicator — by the time the GPU is hot, you're already past the point where scaling decisions should have been made. GPU power draw (watts) is the leading indicator: it reflects current computational load immediately. The NVIDIA Developers Forum threads on GB10 thermal management confirmed that the OEM cooler handles 80-82°C sustained without throttling, meaning the v1.0 thresholds (80°C warning, 83°C critical) were too conservative and caused false triggers. This research produced the power-primary routing model (ADPT-01) where watts are the primary signal and temperature is demoted to a safety brake that only fires at ≥82°C.

**Thread 3: Model-scale-aware batch ceilings.** After the OOM at batch=8, the question was: what batch sizes are actually safe for a 30B model on 128 GB unified memory? The answer couldn't come from this project alone — we'd only tested two configs (batch=4 safe, batch=8 crash). Research across NVIDIA Developers Forum community experience with various model scales on unified memory platforms produced the tiered ceiling model: Small (≤1B) can batch up to 64, Medium (1B-13B) up to 16, Large (13B-30B) up to 8, XL (30B+) capped at 4. These ceilings are conservative by design — the headroom requirements scale with model size because activation memory grows non-linearly with batch size for larger models. For Qwen3-30B-A3B (effective ~27.4B after MoE sparsity), this means the v1.0 batch=8 config was already at the Large-tier ceiling before accounting for the 85% universal safety rule.

**Thread 4: Utilisation vs intensity — the thermal exploitation ladder.** The most surprising research finding was the distinction between GPU utilisation (fraction of time the GPU is working) and intensity (how hard it works per step). The v1.0 planner conflated these — any COOL reading triggered a batch increase, which increases both. But `prefetch_factor`, `save_steps`, and `eval_steps` can increase utilisation (by reducing idle gaps) without touching intensity or memory. The telemetry showed 49-62% GPU utilisation dips between batches (pipeline stalls waiting for data) and 6-7% drops during checkpoint writes (serialisation stalls). These are free wins — zero memory cost, pure pipeline efficiency. The reordered thermal ladder (ADPT-02) exploits this by exhausting all zero-memory optimisations (prefetch, save/eval interval, then workers) before touching the one lever (batch size) that crosses into memory risk territory.

### Externalising to dgx-toolbox

Five capabilities were externalised from this project into dgx-toolbox's new telemetry package (Phase 13):

| Capability | Why External |
|---|---|
| `GPUSampler` (power/temp/util/mem sampling) | Any GPU training project needs this, not just WordPress fine-tuning |
| `AnchorStore` (config history with cooldown + hard caps) | Reusable for any iterative hyperparameter tuning workflow |
| `classify_failure` (NORMAL/OOM/HANG/THERMAL classification) | Every training run on DGX needs failure post-mortem |
| `compute_effective_scale` (param count → tier → batch ceiling) | Model-scale awareness is hardware-specific, not project-specific |
| `probe.py` (warmup probe runner) | Safe batch probing is a generic DGX capability |

The decision criterion was clean: if the capability requires knowledge of WordPress code standards, rubric dimensions, or task tokens, it stays in wp-finetune. If it only requires knowledge of the GPU, the training loop, or the hardware platform, it belongs in dgx-toolbox. GPUSampler doesn't care what model is being trained — it samples `nvidia-smi` metrics. AnchorStore doesn't care what config keys are being tracked — it hashes and persists any dict. The failure classifier reads telemetry patterns (GPU idle + RAM >95% = OOM), not training-specific signals.

This factoring has a practical benefit: when I fine-tune a different model on the same Spark (or a colleague uses theirs), the telemetry infrastructure is already there. The project-specific part — the routing logic, the ladder rung order, the threshold values — stays in `scripts/adaptive_planner.py` and `config/adaptive_planning.yaml`. The generic platform capabilities live in dgx-toolbox where they can evolve independently.

The coupling point is `config/dgx_toolbox.yaml`, which declares the mount (`dgx_telemetry` → `~/dgx-toolbox/telemetry`) and the container PYTHONPATH injection. This keeps wp-finetune ignorant of dgx-toolbox's internal structure — it imports `from telemetry.sampler import GPUSampler` and the mount handles the rest.

### Cross-AI review: what stuck, what was rejected

Both Gemini and Codex reviewed the four plans before execution. Their consensus concerns directly shaped the implementation:

**Accepted — logic in Python, not skill markdown (HIGH from both).** The original plan put the routing algorithm in the skill file. Both reviewers flagged this as untestable. The final implementation has the skill as a thin 7-step orchestration wrapper that calls `scripts/adaptive_planner.py` for every decision. The Python module has 28 unit tests. No algorithm logic lives in markdown.

**Accepted — batch coupling uses `round()` not `//` (HIGH from Codex).** Codex caught that `max(1, effective_batch // new_batch)` doesn't preserve effective_batch for non-divisible changes (batch 4→5 gives accum=3, effective=15 not 16). The implementation uses `round()` and preferentially constrains batch sizes to divisors of effective_batch.

**Accepted — no `sudo drop_caches` in train_model.py (HIGH from Codex).** Both reviewers flagged this as operationally dangerous in containers. Removed entirely — cache management is delegated to dgx-toolbox's UMAMemModel which handles it at the platform level.

**Accepted — no `builtins.print` monkey-patching (HIGH from Codex).** The original plan intercepted Unsloth's banner by wrapping `print`. Both reviewers noted this was fragile and could break logging. The implementation reads `trainer.args` after `build_trainer()` returns — the actual batch/grad_accum values are already set by that point, no interception needed.

**Rejected — Codex's MEDIUM-HIGH overall risk rating.** Codex rated the plans MEDIUM-HIGH overall, while Gemini rated LOW. The difference was focus: Codex evaluated implementation risk of the *original* plan (which had logic in markdown, drop_caches, and print patching), while Gemini evaluated architectural soundness. After the HIGH concerns were addressed in execution, the actual risk matched Gemini's assessment. The lesson: review concerns are most valuable as pre-execution fixlists, not as overall risk scores.

### The three-layer architecture

The final system has three cleanly separated layers:

1. **Platform layer** (dgx-toolbox telemetry/) — Hardware-aware, project-agnostic. Samples GPU metrics, classifies failures, manages config history, runs warmup probes. Evolves with the hardware platform.

2. **Decision layer** (scripts/adaptive_planner.py + config/adaptive_planning.yaml) — Project-specific routing logic. Classifies power zones, applies the thermal ladder, couples batch/grad_accum, computes scale-aware ceilings. Tested with 28 decision-table unit tests. Evolves with training strategy.

3. **Orchestration layer** (adaptive-planner skill + run-training Step 8.5) — Skill-level glue. Reads telemetry, calls decision functions, writes config, manages probe flags between runs. Evolves with the skill system.

Each layer can change independently. Raising the thermal brake from 82°C to 85°C is a one-line YAML edit. Adding a sixth ladder rung is a function addition plus config entry. Replacing GPUSampler with a different sampling backend is invisible to layers 2 and 3.

### Reflection

The v1.0 adaptive planner failed because it modelled the GPU as a single optimisation variable. The v1.1 rewrite succeeds (so far — Plan 04 verification still pending) because it correctly models the two-axis problem: thermal budget is safe to probe, memory budget is not. This asymmetry is structural to unified memory architectures and won't go away.

The externalisation to dgx-toolbox felt like overhead when I started it, but in retrospect it was the forcing function that produced clean abstractions. When you have to explain what GPUSampler does to a different project, you strip away the WordPress-specific assumptions and discover the actual interface boundary. That boundary — "I sample hardware metrics and return numbers; you decide what to do with them" — is the right one.

The 13 requirements for v1.1 (ADPT-01 through PROB-03) are all checked off after three plans. What remains is Plan 04: cross-file integration verification and the human review checkpoint before this goes into a real training run. The planner is code-complete but not battle-tested. Run 4 will be the real validation.

---

## 2026-04-01 — Thermal budget exploitation without memory pressure

### Context

Woke up to discover telemetry shows thermal budget remains unexploited across Run 2 and current Run 3 — GPU averaging 67% utilization at 65-67°C (COOL zone) while the adaptive planner held back from scaling due to the OOM zombie state discovered in Run 2. Run 3 commenced with a conservative configuration (workers=3, batch=4) that stabilized RAM at 97-98 GB with 22 GB headroom, but left significant compute capacity on the table.

The core tension: on DGX Spark's unified memory, compute has a safe observable ceiling (thermal throttling at ~82°C with graceful clock scaling) while memory has an unsafe invisible cliff (driver-level deadlock at ~120 GB with catastrophic system freeze ([pytorch/pytorch#174358](https://github.com/pytorch/pytorch/issues/174358))). You can probe thermal limits freely but you can't probe memory limits safely.

### Research: thermal exploitation options evaluated

Researched six approaches to exploit thermal headroom, cross-referenced against NVIDIA Developers Forum and this project's actual state:

**Already in place:** BF16 (enabled), persistent_workers=True (enabled this session), pin_memory (no-op on Spark's NVLink-C2C — no host-to-device copy to eliminate).

**Adopted — zero/low memory cost:**

| Option | Memory Cost | Rationale |
|---|---|---|
| `prefetch_factor` 2→4 | ~600 MB | Each worker pre-loads more batches into queue. Directly addresses 49-62% GPU dips in telemetry. ~200 MB per worker per increment — negligible. |
| `save_steps` 200→400 | 0 | Telemetry shows GPU drops to 6-7% during checkpoint writes (serializing 3.3 GB adapter weights). Halves these stalls. Watchdog provides safety net between checkpoints. |
| `eval_steps` 100→200 | 0 | Eval runs 5K val examples in inference mode, pausing training. Less frequent eval = more training steps per hour. Loss still logged every 10 steps. |

**Adopted — with model-scale-aware guardrails:**

| Option | Memory Cost | Rationale |
|---|---|---|
| Increase micro-batch (scale-aware) | High — but capped per model size | Not rejected outright — implemented as Rung 4 of the thermal ladder with triple safety gates: model-scale batch ceilings from NVIDIA Developers Forum (XL ≤4, Large ≤8, Medium ≤16, Small ≤64), minimum headroom thresholds that scale with param count (30% for XL), and mandatory warmup probes before any increase takes effect. The 40/60 OOM at batch=8 happened without these guards. With them, batch scaling is available for smaller models while correctly blocked for our 30B where we're already at the XL ceiling. |

**Rejected — memory cost too high for 30B on Spark:**

| Option | Why Not |
|---|---|
| Disable gradient checkpointing | Using `use_gradient_checkpointing="unsloth"` optimized for MoE. Disabling on 30B MoE would add ~20-30 GB activation memory. With 22 GB headroom, instant OOM. |
| torch.compile | 13 GB overhead for ~3% speedup per Spark UGC. Leaves 9 GB headroom — inside the danger zone that killed the 40/60 run. |
| Gradient accumulation | Already at grad_accum=4. Doesn't change per-step intensity — each micro-batch forward/backward is identical. Good for convergence, not for thermal exploitation. |

**Key distinction discovered:** `prefetch_factor` increases *utilization* (fraction of time GPU is working) by reducing micro-stalls between batches. `save_steps`/`eval_steps` increase utilization by reducing macro-stalls (checkpoint writes, eval pauses). `batch_size` increases *intensity* (how hard GPU works per step). Steps 1-2 raise temperature by keeping the GPU busy more often. Step 3 raises temperature by making each step more compute-dense — but only step 3 raises memory.

### Implementation: thermal exploitation ladder in adaptive planner

Replaced the old batch-first scaling policy (Step 8.5e) with a 4-rung prioritized ladder that applies zero-memory changes first:

| Rung | Change | Memory Cost | Trigger |
|---|---|---|---|
| 1 | `prefetch_factor` +1 (cap 4) | ~600 MB | GPU util < 80% |
| 2 | `save_steps` x2 (cap 400) | 0 | GPU util < 80% |
| 3 | `eval_steps` x2 (cap 200) | 0 | GPU util < 80% |
| 4 | `batch_size` +1 (model-scale cap) | High | Rungs 1-3 maxed AND util < 65% AND headroom > scale-aware minimum |

Multiple rungs can fire in a single planning step. The old planner jumped straight to batch/worker scaling and missed the free wins entirely.

### Implementation: model-scale-aware batch ceiling

Rung 4 is now model-scale-aware, using batch ceilings derived from NVIDIA Developers Forum:

| Model Scale | Params | Batch Ceiling | Min Headroom | Notes |
|---|---|---|---|---|
| Small | ≤1B | 64 | 15% of total | I/O bound — batch freely |
| Medium | 1B-13B | 16 | 20% of total | Balanced — room to explore |
| Large | 13B-30B | 8 | 25% of total | Model dominates |
| XL | 30B+ | 4 | 30% of total | At the cliff — batch increase rarely safe |

The 85% memory ceiling rule is the universal safety gate, but larger models need even more margin because activation memory scales non-linearly with batch size. For our Qwen3-30B-A3B, rung 4 is triple-blocked: at ceiling (batch=4 == XL cap), insufficient headroom (17 GB < 36 GB minimum at 30%), and GPU not underutilized enough (67% > 65% threshold).

For a hypothetical 7B model on the same Spark with ~40 GB headroom and 50% util, rung 4 would fire and propose batch 5→6→...→16 incrementally across runs, with warmup probes at each step.

### Implementation: warmup probe for batch scaling safety

The existing watchdog catches gradual memory creep but cannot prevent the Spark's driver-level deadlock on sudden allocation failures — CUDA tries to allocate a large activation tensor, the unified memory driver fails internally, and the system freezes before any Python code executes. The preflight check only runs at startup before the model is loaded.

When rung 4 fires (batch+1), the planner writes a `_warmup_probe_required` flag. Before the next run's full training starts, Step 6 runs 1 real training step at the new batch size and checks `/proc/meminfo`. If the probe survives, proceed. If it OOMs or memory drops below 2 GB available, revert to the previous batch size automatically. Cost of a failed probe: 1 step (not hours).

### Implementation: prefetch_factor passthrough

Added `dataloader_prefetch_factor` passthrough to `train_model.py` → `SFTConfig`. Verified it's accepted by HuggingFace Trainer (inherits from PyTorch DataLoader kwargs) and requires only `num_workers > 0` (we have 3-4).

### Adaptive resource planning summary

The adaptive planner now manages two resource budgets with fundamentally different risk profiles:

**Thermal budget** (safe to probe): GPU temperature and utilization. Observable via telemetry, degrades gracefully via clock scaling at ~82°C, recoverable. The thermal exploitation ladder pushes utilization from ~67% toward 75-80% through pipeline optimization (prefetch, fewer stalls) before considering compute intensity increases (batch size).

**Memory budget** (unsafe to probe on Spark): Unified memory usage. The DGX Spark's driver-level deadlock on allocation failure means there's no safe way to discover the memory limit at runtime — you either stay below it or the system freezes. The planner enforces model-scale-aware ceilings derived from community experience, requires warmup probes before any batch increase, and applies the 85% universal ceiling rule.

The asymmetry is structural: compute has a safe, observable ceiling with graceful degradation. Memory has an unsafe, invisible cliff with catastrophic failure. The ladder respects this by exhausting all zero-memory thermal optimizations before touching the one lever (batch size) that crosses into memory risk.

### Reflection

Finetuning 30B models on 120 GB unified memory is operating in a narrow corridor: ~97 GB floor (model + optimizer + activations) to ~102 GB ceiling (85% safety rule). That's a 5 GB optimization window. The thermal ladder exploits what's available by reducing idle time, but it can't change the fundamental geometry that the model consumes 80%+ of memory before training even starts. The remaining ~30% GPU idle time isn't wasted thermal budget — it's the cost of the safety margin that keeps the system alive.

---

## 2026-03-31 — Run 2 OOM crash, DGX Spark unified memory deadlock, config rollback

### What happened

Run 2 (40/60 split) crashed twice from OOM — not a clean CUDA OOM, but the DGX Spark's unified memory driver-level deadlock. The GB10's shared memory pool (CPU + GPU + page cache in ~128 GB) has no discrete VRAM, so when memory pressure spikes, the NVIDIA driver's internal descriptor allocations fail below the CUDA runtime layer. The GPU context cascades into failure, `nvidia-modeset` enters uninterruptible D-state, and the entire system freezes (SSH dies, no logs flush). This is a known open issue on NVIDIA's tracker — the driver doesn't handle reclaim latency gracefully on UMA platforms.

### Telemetry timeline

The telemetry agent captured the full progression across 85 samples at 10-min intervals:

- **20:51 UTC** — Pre-training idle at 9.4 GB (8% RAM)
- **21:01** — Model loaded, training starts: 95.4 GB (80%), GPU 82%, 65C
- **21:21** — Steady state: ~103 GB (87%)
- **01:21** — Memory creep begins: 111 GB (93%) — sawtooth pattern from dataloader worker respawn
- **04:51** — **First OOM**: RAM hit 119.3 GB (97%), GPU 95%, temp 79C — peak on both axes
- **05:01** — OOM kill fires, RAM drops to 92 GB, training auto-restarts from checkpoint
- **05:11–09:51** — Second attempt oscillates 99–121 GB, same sawtooth creep
- **10:01** — **Second OOM**: 122.5 GB (99.9%), GPU drops to 5% — training killed again
- **10:11–10:41** — Dead state: GPU idle (4–7%), RAM pinned at 122 GB ceiling, training did not recover
- **10:51** — Monitor stopped (`max_checks_reached`)

Last valid checkpoint: `checkpoint-2200` at step 2200/5084 (epoch 0.87).

### Root cause

The adaptive planning after Run 1 doubled `dataloader_num_workers` from 4→8 and `per_device_train_batch_size` from 4→8 based on Run 1's 15.7 GB headroom and COOL thermal zone. This was too aggressive — Run 2 has ~17% more training data (40/60 vs 30/70), and 8 workers each prefetch batches with 4096 seq_length into RAM. The sawtooth pattern (workers killed → respawned → re-allocate buffers) caused periodic spikes that hit the memory ceiling.

Run 1 comparison: peaked at 104 GB with workers=4, batch=4 — 15.7 GB headroom, no OOM.

### Why discrete GPUs don't have this problem

On a discrete GPU, PyTorch gets a clean `RuntimeError: CUDA out of memory` from the CUDA runtime and can handle it gracefully. On the Spark, the failure occurs in the driver's internal allocation path (NV_ERR_NO_MEMORY), below where CUDA would catch it. The process never gets a chance to throw the Python-level exception — instead the whole system deadlocks. Hard reboot is the only recovery.

### Additional memory creep mechanisms (from research)

- **PyTorch caching allocator fragmentation** — variable-sized tensor alloc/free over thousands of steps creates increasingly fragmented memory; reported up to 43% waste in large-scale LLM training
- **torch.compile() host RAM leak** — internal caching of compiled graph artifacts grows with every minibatch (not using this, but noting for future)
- **Checkpoint spikes** — serializing full model + optimizer state temporarily doubles memory usage
- **DataLoader worker accumulation** — each `num_workers > 0` subprocess gradually grows resident memory from Python object overhead

### Config changes made

Rolled back the aggressive adaptive scaling and added persistent workers to eliminate the sawtooth respawn pattern:

```yaml
# config/train_config.yaml
per_device_train_batch_size: 4    # was 8 (adaptive) — back to Run 1 proven value
gradient_accumulation_steps: 4    # was 2 (adaptive) — keeps effective batch=16
dataloader_num_workers: 6         # was 8 (adaptive) — split difference: more GPU util than 4, less RAM than 8
dataloader_persistent_workers: true  # NEW — workers stay alive between epochs, eliminates respawn allocation spikes
```

Also added `dataloader_persistent_workers` passthrough in `scripts/train_model.py:295` to SFTConfig.

### Mitigations — what was done, what was skipped

The three implemented mitigations form layered defense in depth — they operate at different timescales and catch different failure modes:

| Layer | When | Catches | Without it |
|-------|------|---------|------------|
| Peak-based headroom (8.5a/e) | Between runs (planning) | Prevents OOM by setting conservative config based on prior peak RAM | Next run starts with a config that will OOM again |
| OOM-aware adaptive planner (8.5c/d-mem) | Between runs (post-mortem) | Detects that an OOM already happened, backs off harder than thermal scaling would | After a crash, the planner sees COOL thermals and scales back up to the config that just crashed |
| Memory watchdog callback | During training (every step) | Unpredicted memory creep — fragmentation, dataloader accumulation, checkpoint spikes | Training crashes at 99.9% RAM with no checkpoint, losing hours of work |

The first two are **preventive** — they configure the next run to avoid OOM. The watchdog is **reactive** — it saves the current run before it dies. Even with perfect between-run planning, memory can creep unpredictably mid-run (the telemetry showed a sawtooth pattern from worker respawns, 10–20 GB above the mean). The watchdog is the only layer that protects in-flight work.

**Implemented:**

- **Memory watchdog callback** (`scripts/train_model.py`) — `MemoryWatchdogCallback` reads `/proc/meminfo` every training step. When `MemAvailable` drops below 2 GB, it sets `should_save = True` and `should_training_stop = True`, triggering a clean checkpoint save before the OOM killer strikes. Fail-open: if `/proc/meminfo` can't be read, it returns a high value and never triggers. The 2 GB threshold leaves room for the checkpoint save itself (~1.2 GB for the LoRA adapter + optimizer state). This prevents the scenario from Run 2 where up to 200 steps of training were lost per OOM kill.

- **Adaptive planning fixes** (Step 8.5 of `/run-training` skill) — all three issues from the root cause analysis were fixed:
  - **8.5a** — Added peak RAM tracking with 5 GB safety margin (`effective_headroom_gb`), plus OOM detection from telemetry (GPU idle + RAM >95% in final readings)
  - **8.5b** — Thermal history records now include `peak_ram_gb`, `p95_ram_gb`, `safe_headroom_gb`, `effective_headroom_gb`, `likely_oom`, and `dataloader_persistent_workers`
  - **8.5c** — OOM overrides thermal: if `likely_oom` is True, skip thermal scaling entirely and jump to memory backoff
  - **8.5d-mem** (new step) — Memory backoff: restores last non-OOM config, steps workers down by 1, force-enables `persistent_workers`
  - **8.5e** — Uses `effective_headroom_gb` (peak-based + 5 GB margin) instead of average-based headroom. Batch cap lowered from 16 to 8 on unified memory. Workers scale +1 at a time (not doubling), hard-capped at 6 on unified memory, decrease if headroom <10 GB. `persistent_workers` is preserved once enabled
  - **8.5f** — Writes `dataloader_persistent_workers` to config, logs OOM count and effective headroom

**Skipped — `torch.cuda.empty_cache()` before checkpointing:**

On unified memory (DGX Spark), the CUDA allocator and system RAM share the same pool. `empty_cache()` releases the allocator's free blocks, but those are typically small fragments — not the large contiguous allocations that cause OOM. Worse, it forces reallocation on the next forward pass, which can actually increase peak memory momentarily and hurt throughput. The real memory pressure comes from dataloader workers and optimizer states, not CUDA cache fragmentation.

**Still to investigate for future runs:**

- Drop page cache before training: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
- Disable swap: `sudo swapoff -a` (swap extends thrashing window before OOM killer can act on UMA)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces fragmentation by growing segments rather than allocating new fixed-size blocks
- Docker memory limits (`--memory=90g`) as additional safety layer (NVIDIA forums report these don't fully contain the issue)
- OOM score override for SSH: `OOMScoreAdjust=-1000` (doesn't prevent driver deadlock but protects SSH if OOM killer fires first)

### Lesson for adaptive planning

The adaptive convergence loop assumed monotonic scaling was safe if thermal zone was COOL. It didn't account for the memory dimension separately from temperature. On unified memory architectures, memory pressure can hit the ceiling while thermals are still comfortable. The adaptive system now checks memory pressure alongside thermal zone classification — using peak RAM with a safety margin, not average — and has OOM-aware backoff that restores the last known-safe config.

### How the adaptive planning caused this

The adaptive resource planning (Step 8.5 of the `/run-training` skill) optimised for thermal budget but had no concept of memory pressure. After Run 1, it saw COOL zone with 32% avg GPU idle and scaled up:

```
### After ratio 30_70
- batch_size: 4 → 8
- grad_accum: 4 → 2
- workers: 4 → 8
- reason: COOL zone with 32% avg GPU idle — scaling up to fill headroom
```

This is exactly what caused the OOM. The adaptive logic overwrote `train_config.yaml` between runs, and the new config exceeded the Spark's unified memory ceiling. The headroom calculation used **average** RAM, not **peak** spikes from worker respawns — on unified memory, the peak is what kills you. This has now been fixed (see above).

### Resume gap in `/run-training` skill

Attempted to resume the 40/60 run from checkpoint-2200 using `/run-training`, but the skill had no resume path. The idempotency check in Step 7 correctly identified the run as incomplete (no `adapter_config.json`), but the training command was hardcoded as a fresh start — `python -m scripts.train_model --config ...` with no `--resume` flag. This meant re-running the skill would restart training from step 0, discarding the 2200 steps already completed (~7 hours of GPU time).

**Fix applied to Step 7b:** The skill now scans `adapters/{run_name}/` for `checkpoint-*` directories before launching training. If checkpoints exist, it sorts them numerically, finds the latest, and appends `--resume <path>` to the train command. If no checkpoints exist, it runs fresh as before. The idempotency check (`adapter_config.json`) still gates completed runs — only interrupted runs with checkpoints but no final adapter get the `--resume` treatment.

This was a gap in the original skill design: it handled completed runs (skip) and fresh runs (start) but not the middle case — partially completed runs that should resume.

---

## 2026-03-31 — Run 2 commencing (40/60), adaptive resource planning working

### Run 2: 40/60 ratio

Starting the 40/60 ratio — the one I expect to yield the best results for both generation and judging. The original project spec called for 40/60, and now with 30,498 judge examples it's no longer a compromise forced by data scarcity. This ratio gives the judge pathway 60% of training time (more rubric scoring practice) while still providing substantial generation examples (20,332). If any ratio produces genuinely bipolar behaviour through `<wp_gen>` and `<wp_judge>` task tokens, this is the one.

### Adaptive resource planning is working

The system I built in the previous session actually works. After Run 1 completed, the orchestrator automatically:

1. Parsed the telemetry from the 30/70 run (85 readings)
2. Classified the thermal zone: **COOL** (peak 71C, avg 65.3C)
3. Confirmed the config bump to batch=8, grad_accum=2, workers=8 was appropriate
4. Wrote the thermal history to `telemetry/training/thermal_history.json` (1 run recorded)
5. Logged the adjustment to `telemetry/training/adaptive_adjustments.md`

The COOL zone (65-71C) with 68% avg GPU utilisation confirms there's still headroom. The config was already bumped from the conservative batch=4 to batch=8 based on the underutilisation observation, and the thermal data validates that decision — the GPU isn't stressed. If Run 2 pushes into WARM (72-77C), that's the target zone and the config will hold. If it stays COOL, the system will try to scale up further for Run 3.

This is the convergence loop in action: Run 1 data → classify → adjust → Run 2 → repeat. No manual tuning needed between runs.

---

## 2026-03-31 — Run 1 complete overnight, NVML stale context false alarm

### What happened

Woke up to the first training run (30/70) completed — 4,358/4,358 steps, epoch 2.0, final loss ~0.29, 3.4GB adapter saved with 22 checkpoints. But the container had lost GPU access (NVML error). Needed a container restart to re-bind the GPU before starting run 2.

### Investigation

Four possible causes for a container losing GPU access:

1. **NVIDIA driver reload** — host driver updated/reloaded, container's `/dev/nvidia*` handles become stale
2. **GPU reset** — thermal event, ECC error, or `nvidia-smi -r` invalidates the NVML context
3. **cgroup changes** — systemd or Docker daemon restart revokes device cgroup permissions
4. **Suspend/resume** — host suspended (unlikely on a server, but DGX Spark is a desktop form factor)

Diagnosed as **#1 — stale container with long-lived GPU context**. The evidence:

- `dmesg` had zero NVIDIA/NVRM/XID entries — the driver never crashed or reset
- No system suspend events — only `cups-browsed` hourly sleep entries. Uptime was 14 days continuous
- Docker daemon never restarted
- Container ran for **41 hours** (`execDuration=41h2m53s`) before being force-killed (exit 137 = SIGKILL)

On the GB10's unified memory architecture, long-lived NVML contexts can become stale. The host-side driver manages unified memory differently than discrete GPU memory. After extended training with repeated model loads, the container's NVML library loses its ability to query GPU state, even though the GPU itself is fine (host `nvidia-smi` showed 48C, 13W, 7% util — perfectly healthy).

### Resolution: false alarm, no fix needed

The NVML context went stale **after** training completed, not during. The container was sitting idle for hours waiting for the next run to be triggered. Training data and adapters are fully intact.

Since the telemetry agents already run `nvidia-smi` on the host (not inside the container), this doesn't affect monitoring. The orchestrator restarts the container between runs anyway.

### Note for others

If you're monitoring GPU health with `nvidia-smi` **inside** a container deployed for long-lived training runs — particularly on unified memory architectures like the GB10, or in environments where you can't install `nvidia-smi` on the host machine — be aware that the NVML context can go stale after 24+ hours. The GPU is fine; the container just can't see it anymore. A container restart fixes it. Consider adding a periodic `nvidia-smi` health check inside the container and triggering a checkpoint-save + restart if it fails.

---

## 2026-03-29 — Adaptive resource planning: telemetry-informed convergence toward thermal sweet spot

### Context

After catching the GPU underutilisation on the first run (35% avg util with batch=1), I pushed to a more aggressive config (batch=4, 4 workers, gradient checkpointing). This improved utilisation but still left headroom — the GPU was running comfortably in the COOL zone (65-71C). Rather than manually tuning after each run, I built an adaptive feedback loop between the telemetry agents and the training orchestrator.

### The convergence function

The `/observe-training` telemetry agents collect GPU temperature, utilisation, and VRAM usage during each run. Between runs (Step 8.5 of the training skill), the orchestrator reads these metrics and classifies the thermal zone:

| Zone | Temp Range | Action |
|------|-----------|--------|
| CRITICAL | >= 83C peak | **Pause.** Backoff to last WARM config. Alert user. Wait for cooldown. |
| HOT | 78-82C | Step down: reduce batch by 1, increase grad_accum to compensate |
| WARM | 72-77C | **Target zone.** Hold config — no changes needed |
| COOL | 65-71C | Scale up if GPU util < 75% and VRAM headroom > 10GB |
| COLD | < 65C | Aggressive scale-up if GPU util < 60% |

The system converges toward the WARM zone (72-77C) across sequential runs — each run informs the next. Expected convergence:

```
Run 1: COLD (batch=4)  → scale up to batch=8
Run 2: WARM (batch=8)  → hold (saved as safe point)
Run 3: COOL (batch=8)  → scale up to batch=12
Run 4: CRITICAL (batch=12) → backoff to batch=8 (last WARM)
Run 5: WARM (batch=8)  → hold (system stabilised)
```

### Thermal history as memory

The key insight: the system needs to remember what worked. A persistent `telemetry/training/thermal_history.json` records each run's config and thermal outcome. When a CRITICAL event occurs, the backoff doesn't blindly halve the batch size — it restores the exact config from the last run that was in the WARM zone. This is the last known-safe operating point.

If no WARM history exists (first run overheats), it falls back to halving as a conservative default. The history file persists across skill invocations and context resets.

### Live thermal guard

During training, the telemetry observer touches `_thermal_pause` if GPU hits >= 83C. The orchestrator checks for this after training ends and applies CRITICAL backoff rules before starting the next run. This prevents multi-day sequential runs from cooking the GPU if conditions change (ambient temperature, other workloads).

### Telemetry is now default-on

Changed Step 0c of the training skill: telemetry is enabled by default because adaptive resource planning depends on it. If the user tries to disable, a warning explains the consequence (no auto-adjustment, risk of underutilisation or overheating) and requires double-confirmation.

### Why this matters

With 5 sequential training runs (30-60 hours total), I can't babysit the GPU for the entire duration. The adaptive loop means the system self-tunes: if the first run runs cold, the second run pushes harder; if a run gets too hot, it backs off to the last safe config. By the third or fourth run, the config should have converged to the thermal sweet spot for this specific model on this specific hardware.

This is another instance of the outcomes-driven pattern: define the desired state (WARM zone), define the feedback mechanism (telemetry), define the recovery logic (backoff to last WARM), and let the system converge.

---

## 2026-03-29 — GPU underutilisation: conservative config wasting DGX Spark capacity

### What the telemetry showed

The first training run (30/70 ratio) was running with telemetry enabled. The observers caught a problem I wouldn't have noticed for hours otherwise:

| Time | GPU Util | Temp | Notes |
|------|----------|------|-------|
| 05:05-05:35 | 4-8% | 54-57C | Model loading / tokenisation |
| 05:45 | 77% | 63C | Brief spike (torch.compile warmup) |
| 05:49-06:39 | 4-53% | 60-63C | Training steps — oscillating, avg ~35% |

The GPU was averaging ~35% utilisation during actual training. At 8W power draw and 60-63C, the DGX Spark was barely working. This is a 128GB unified memory machine with a Blackwell GB10 — it should be saturated during training, not idling between steps.

### Root causes

1. **Batch size 1 with grad_accum=8** — the GPU processes one example at a time, then idles during gradient accumulation. Effective batch is only 8, but the GPU only sees 1 example per forward/backward pass.
2. **~50 seconds per step** — most time spent on CPU-side data loading/processing between GPU passes. Zero DataLoader workers meant sequential data loading.
3. **No gradient checkpointing in SFTConfig** — while Unsloth sets it on the model side, the trainer config wasn't using it, limiting how much batch size could increase.
4. **Massive memory headroom** — the 63GB model leaves ~65GB unused. No reason to be conservative with batch size.

### Fix: aggressive config

Stopped the first run, updated `config/train_config.yaml`:

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| `per_device_train_batch_size` | 1 | 4 | GPU processes 4 examples per pass instead of 1 |
| `gradient_accumulation_steps` | 8 | 4 | Effective batch stays at 16 (4x4), but GPU works 4x harder per step |
| `gradient_checkpointing` | not set | true | Enables larger batch by trading compute for memory |
| `dataloader_num_workers` | 0 | 4 | Parallel data loading eliminates CPU-side idle time |

Regenerated all 5 per-ratio config overlays and cleaned the partial 30/70 adapter. Restarting from scratch with the new config.

### Lesson learned

The telemetry framework paid for itself on the first training run. Without the GPU metrics observer, I would have let it run at 35% utilisation for 60+ hours — effectively wasting 65% of the compute. The fix took 5 minutes; the time savings across 5 runs is substantial.

Conservative defaults are safe for a first run, but always check utilisation early. A 5-minute telemetry review after the first 100 steps would have caught this immediately.

---

## 2026-03-29 — Training commenced: 5 sequential LoRA runs, ~30-60 hours estimated

All 5 ratio variants are now running sequentially on DGX Spark. This is the moment the dataset work pays off — or doesn't.

### Training plan

| Property | Value |
|----------|-------|
| Base model | Qwen/Qwen3-30B-A3B (downloaded, verified) |
| LoRA config | r=32, alpha=64, dropout=0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_up_proj, down_proj |
| Modules saved | embed_tokens, lm_head |
| Training | 2 epochs, batch=1, grad_accum=8 (effective batch 8) |
| Learning rate | 2e-4, cosine scheduler |
| Precision | BF16 |
| Telemetry | Enabled (`/observe-training` + `/review-telemetry`) |
| Disk available | 3.3TB (8% used) |
| Memory required | ~70GB (BF16 model + LoRA optimizer states) |

### 5 sequential runs

| Run | Dataset | Gen | Judge | Train examples | Output |
|-----|---------|-----|-------|----------------|--------|
| 1 | ratio_30_70 | 13,071 | 30,498 | 34,855 | `adapters/qwen3-30b-wp-30_70/` |
| 2 | ratio_40_60 | 20,332 | 30,498 | 40,664 | `adapters/qwen3-30b-wp-40_60/` |
| 3 | ratio_50_50 | 30,498 | 30,498 | 48,796 | `adapters/qwen3-30b-wp-50_50/` |
| 4 | ratio_60_40 | 45,747 | 30,498 | 60,996 | `adapters/qwen3-30b-wp-60_40/` |
| 5 | ratio_70_30 | 71,162 | 30,498 | 81,328 | `adapters/qwen3-30b-wp-70_30/` |

Estimated 6-12 hours per run depending on dataset size, ~30-60 hours total. Each run produces an isolated adapter in its own directory. Same base model, same hyperparameters, only the gen/judge ratio changes — a clean A/B/C/D/E test.

After all 5 complete, each adapter gets evaluated against the 193-check canonical rubric via `eval_gen.py`, `eval_judge.py`, and `eval_gate.py`. The data decides the optimal ratio.

---

## 2026-03-29 — Milestone: judge pool surpasses CoT reasoning, 4-way split dataset

### The breakthrough

The judge data bottleneck is gone. Full-coverage judge training combined with a 4-way CoT split has transformed the dataset from ratio-constrained to ratio-flexible:

- **Judge pool:** 3,956 → 30,498 (7.7x increase)
- **CoT data:** 610 → 29,020 (47x increase across 4 specialised types)
- **Total dataset:** 5,868 → up to 101,660 depending on ratio

### 4-way CoT split

Instead of treating CoT as a single category, I split it into four types that each teach a different reasoning pathway:

| Type | Source | Teaches |
|------|--------|---------|
| Gen: Pattern CoT | Passed functions | "Requirement → pattern → implementation → reasoning" |
| Judge: Rubric CoT | Mixed passed+failed | "Code → walk through 9 dimensions → scores → verdict" |
| Judge: Contrastive CoT | Failed functions | "Bad code → issues → fixes → what good version looks like" |
| Shared: Security CoT | Functions with security surface | "Security analysis → nonce/cap/escape → verdict" |

Each split has a 10% minimum floor and 500-example minimum to prevent any pathway from being starved.

### Final ratio exports

All 5 ratio variants exported to `data/final_dataset/ratio_{gen}_{judge}/`:

| Ratio | Gen | Judge | Total | Train |
|-------|-----|-------|-------|-------|
| 30/70 | 13,071 | 30,498 | 43,569 | 34,855 |
| 40/60 | 20,332 | 30,498 | 50,830 | 40,664 |
| 50/50 | 30,498 | 30,498 | 60,996 | 48,796 |
| 60/40 | 45,747 | 30,498 | 76,245 | 60,996 |
| 70/30 | 71,162 | 30,498 | 101,660 | 81,328 |

The judge count holds steady at 30,498 across all ratios — the constraint has flipped from "not enough judge data" to "how much generation data to include." This is a much better problem to have.

---

## 2026-03-28 — Observation: good/bad code imbalance and dataset ratio strategy

### The imbalance

Using top plugins and themes by active installs as the source corpus has created a disproportionate ratio of "good" to "bad" code units. First pass results:

| Category | Count | Use |
|----------|-------|-----|
| Real code (passed judge) | 22,137 | `<wp_gen>` generation examples |
| Synthetic (passed) | 2,720 | `<wp_gen>` generation examples |
| Judge training (scored) | 3,956 | `<wp_judge>` critique examples |

That's roughly **7:1 good-to-bad**. I probably should have expected this. The most popular WordPress plugins and themes tend to be well-maintained, well-reviewed code — they're popular *because* they're good. Selecting by active installs was a deliberate quality filter for the generation pathway, but it systematically starved the judge pathway of the negative examples it needs to learn from.

### The ratio enforcement problem

The export pipeline enforces a 40/60 gen/judge split. With only 3,521 judge examples, this caps the total dataset:

- Pre-dedup merged: 29,423 examples
- After dedup: 26,915
- After 40/60 ratio enforcement: **5,868** (2,347 gen / 3,521 judge)

We're throwing away ~21,000 generation examples to maintain the ratio. The bottleneck isn't data volume — it's judge data scarcity.

### Plan: test multiple ratios via evals

Rather than committing to one ratio, I'm going to export multiple versions and compare eval scores after training:

- **40/60** (current) — 5,868 examples, emphasises judge capability
- **50/50** — ~7,000 examples, balanced
- **60/40** — ~8,800 examples, prioritises code generation (the primary use case)
- **Uncapped with 2K judge floor** — ~27,000 examples, uses all available data

The 40/60 split was set before I knew the actual data distribution. Now that I know gen outnumbers judge 7:1, it's worth testing whether the judge pathway really needs 60% of training time or whether 3,500 well-scored examples is sufficient for 0.85 Spearman correlation.

### Expanding the source corpus — both directions

**More good code (generation):** 24 plugins with GitHub URLs not yet in `repos.yaml` — including Elementor (10M installs), WooCommerce (7M), wpforms-lite (6M), Jetpack (3M), and Rank Math (3M). Adding these could yield 5,000-15,000 additional extractable functions. Three block themes (twentytwentyfour, twentytwentythree, twentytwentytwo) also available for modern Full Site Editing patterns.

**More bad code (judge):** I need to pull a separate dataset of poorly rated, vulnerable, out-of-date, and unscalable plugins and themes. The current corpus is biased toward quality by design — to train a judge that can identify *bad* code, I need to show it real-world bad code, not just synthetic mutations. Candidates: plugins with known CVEs (CVSS > 7), plugins not updated in 2+ years, plugins with < 3-star ratings, abandoned themes with known compatibility issues. This is a different curation exercise than the original "top by installs" approach — deliberately selecting for poor quality rather than filtering it out.

I've added the 24 new plugins and 3 themes to `repos.yaml` and am re-running the full data pipeline to process all repos. I'll run evals with the current dataset in parallel, but will also enrich the judge training data with poor code examples from the bad-code corpus as a follow-up.

### Side observation: Claude Code agents vs API-driven scripts for data pipeline work

Claude Code agents are significantly faster at judging and CoT reasoning across code units than the static Anthropic API-driven scripts I built in Phase 1. The original `phase1_judge.py` and `phase2_judge_dataset.py` scripts used the Batch API with polling loops, retry logic, JSON extraction fallbacks, and checkpoint management — all of which I had to build and debug myself. Claude Code agents handle all of that natively: they read the rubric, read the code, produce structured output, and move on. No polling, no retry wrappers, no batch ID tracking.

The speed difference isn't just API latency — it's developer velocity. Writing a prompt for a Claude Code agent takes minutes. Writing, testing, and hardening an API-driven script with the same capability took hours. And the agent can adapt mid-run (read a different file format, handle an edge case) without a code change.

**Recommendation for future projects:** Use Claude Code agents as first-class integration for any LLM-driven pipeline work. Reserve direct API calls for cases where you need deterministic, repeatable outputs with exact token control (e.g., structured evaluation benchmarks). For data processing, judging, generation, and CoT — agents are faster to build, faster to run, and cheaper (subscription vs per-token).

### Update: pulling all repos confirmed the quality bias

I pulled ALL repos from `repos.yaml` (not just the ones already processed) in an attempt to enrich the judge dataset. Result: top plugins and themes are just generally coded to a high standard. More repos produced more generation examples but barely moved the judge needle — the imbalance worsened to ~14:1 gen-to-judge.

Current state across all ratio options:

| Ratio | Gen | Judge | Total |
|-------|-----|-------|-------|
| 40/60 (current) | 2,347 | 3,521 | 5,868 |
| 50/50 | 3,521 | 3,521 | 7,042 |
| 60/40 | 5,282 | 3,521 | 8,803 |
| 70/30 | 8,216 | 3,521 | 11,737 |
| Uncapped | 48,333 | 3,521 | 51,854 |

The judge count is stuck at 3,521 regardless of how many repos I process. The generation count scales freely (48K uncapped) but is meaningless without matching judge data.

### Decision: deliberately curate a poor-code corpus

Since the top-plugins approach won't produce bad code no matter how many repos I add, I did the same discovery process but inverted the selection criteria: plugins and themes rated 3 stars or less on WordPress.org, with at least 100 active installations, ordered by most active installs descending. The goal was to find poorly implemented code for the judge pathway's negative dataset.

**Data collection process:**

Queried the WordPress.org API for both plugins and themes. The plugin directory yielded 1,000 qualifying entries after scanning 16,500 plugins (66 pages). The themes directory is far more curated — across all 8,007 themes, only 186 had ratings of 3 stars or less with 100+ installs. That's the hard ceiling, not a sample.

Enriched all entries with vulnerability data from WPVulnerability.net (CVSS scores, unpatched vulns, CWE classifications). Then ran a three-phase GitHub URL discovery process: WordPress.org page scraping (356 repos), `gh search repos` CLI search (501 repos), followed by a validation pass to classify official developer repos vs mirror repos (wp-plugins/, WordpressPluginDirectory/, etc.) and remove false positives. Notable catches: Gutenberg itself at 2.1 stars (3,863 ratings, 300K installs), WooCommerce PayPal Payments at 2.7 stars, Meta for WooCommerce at 2.2 stars.

**Results — 4 datasets total:**

| Dataset | Rows | GitHub URLs | Official | Mirror |
|---------|------|-------------|----------|--------|
| Top 1000 plugins | 1,000 | 776 (77.6%) | 561 | 215 |
| Top 100 themes | 100 | 25 (25%) | 25 | 0 |
| Poor plugins (<=3 stars) | 1,000 | 163 (16.3%) | 153 | 10 |
| Poor themes (<=3 stars) | 186 | 1 (0.5%) | 1 | 0 |

Coverage varies predictably: top plugins maintain public GitHub repos; themes mostly use WordPress.org's SVN system; poorly-rated projects tend to be smaller with no public source repos. Mirror repos still contain the actual code (synced from SVN) so they're usable for extraction.

### Hypothesis confirmed: poor-code corpus transformed the dataset

After running the full pipeline on all repos including the poor-code corpus, the judge bottleneck broke:

| Ratio | Gen | Judge | Total |
|-------|-----|-------|-------|
| 40/60 (current) | 11,926 | 17,889 | 29,815 |
| 50/50 | 17,889 | 17,889 | 35,778 |
| 60/40 | 26,833 | 17,888 | 44,721 |
| 70/30 | 41,740 | 17,888 | 59,628 |
| 80/20 | 71,556 | 17,888 | 89,444 |
| Uncapped | 76,741 | 17,889 | 94,630 |

**Before:** 3,956 judge examples bottlenecked everything at ~6K total. **After:** 17,889 unique judge examples — even at 40/60, I get 29,815 examples. The dataset is no longer ratio-constrained at any practical split. The poorly-rated repos contributed exactly what I needed: organic defects (sloppy SQL concatenation, missing nonces, unescaped output in legacy admin pages) that the 9-dimension rubric scores low.

### Plan: export all ratios and experiment

The concrete plan:

1. Export at 30/70, 40/60, 50/50, 60/40, 70/30
2. Train on each — LoRA is cheap enough to run 5 times
3. Eval each against the canonical rubric in `docs/eval/`, not just the Phase 4 gate thresholds
4. Let the data decide

The eval infrastructure has three layers:

**Canonical rubric** (`docs/eval/wp_code_quality_rubric.md`) — 193 unique check IDs (83 positive signals, 110 negative signals) across all 9 weighted dimensions, each with a detection method and point weight. Security is weighted 20%, SQL Safety 15%, with critical floor rules: if a direct XSS vector is found, Dimension 2 cannot score above 3/10 regardless of other patterns. The rubric defines a ground truth scoring procedure using a multi-tool pipeline — PHPCS with WordPress/VIP/Security standards (~120 check IDs), PHPStan level 5 with wordpress-stubs, regex patterns for 30+ checks not covered by PHPCS (N+1 loops, missing transient caching), and LLM judgment for 18 checks requiring semantic understanding (architectural appropriateness, ARIA completeness).

**Research backing** (`docs/eval/research_wpcs_standards.md`, `docs/eval/research_wp_security_sql_perf.md`) — complete evidence base: all 58 WPCS sniffs, 39 VIP sniffs, Plugin Check tool checks, PHPCompatibilityWP rules, full sanitization/escaping function tables, `$wpdb` method reference, N+1 detection patterns, 18 translation functions, 21 accessibility criteria, hook anti-patterns, REST API permission/schema requirements.

**Eval scripts** (`eval/eval_gen.py`, `eval/eval_judge.py`, `eval/eval_gate.py`) — `eval_gen.py` measures generation quality (PHPCS pass rate, security pass rate against the rubric's check IDs), `eval_judge.py` measures scoring correlation (Spearman against the multi-tool ground truth, not just PHPCS alone), `eval_gate.py` applies pass/fail thresholds.

Five LoRA training runs with the same base model, same hyperparameters, only the ratio changes — a clean A/B/C/D/E test evaluated against the full 193-check rubric.

My concern remains that an MoE model with 128 experts and top-8 routing may not provide enough separation between expert pathways to produce genuinely "bipolar" behaviour — harshly judging poorly created code via `<wp_judge>` while still generating high-quality output via `<wp_gen>`. The experts were pre-learned on general data, not task-specific routing, so the fine-tuning has to create that separation through the task token signal alone. Five ratio variants let me test whether more judge data actually improves the judge pathway's discrimination, or whether it plateaus and the extra data just adds noise.

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

These skills are the same thing. The skill file is the assembly instructions. The YAML config is the bill of materials. The Python engine is the Allen key. The agents are the person. They don't get frustrated, they don't misread the instructions (though they may misinterpret them — an important distinction), and they don't call customer support to complain.

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
