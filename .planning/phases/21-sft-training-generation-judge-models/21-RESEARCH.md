# Phase 21: SFT Training — Generation & Judge Models - Research

**Researched:** 2026-07-13
**Domain:** Cloud LoRA SFT (Tinker) on a new MoE base (Qwen/Qwen3.6-35B-A3B), reusing existing training/eval data and the exact v1.2 (gen) / v1.3 (judge) recipe from the old base (Qwen3-30B-A3B)
**Confidence:** MEDIUM-HIGH (repo mechanics VERIFIED by reading the actual scripts/receipts; new-base deltas inherit the Phase 19/20 research confidence levels)

## Summary

Phase 21 does NOT invent a new training method. Its entire job is: take the exact Tinker SFT driver
(`scripts/tinker_reasoning_sft.py`), the exact data adapter (`scripts/tinker_reasoning_data.py`), the exact
eval scripts (`scripts/relabel/eval_relabel.py`, `scripts/capture_judge_responses_tinker.py`,
`scripts/run_eval_reasoning.py`), and the exact reused datasets (`data/reasoning_dataset/`,
`data/relabel_v1/`) that produced v1.2 (gen, wp-bench 0.4484) and v1.3 (judge, ensemble rho 0.8075) on
Qwen3-30B-A3B, and re-point them at `Qwen/Qwen3.6-35B-A3B`. Every one of these scripts currently hardcodes
the OLD base (`BASE_MODEL = "Qwen/Qwen3-30B-A3B"`) and must be parameterized or forked for v4 — none of
this has been done yet; Phase 20 only proved the *merge/serve* path (attention-only probe adapter), not the
*train* path.

The single highest-risk gap this research surfaces: **`merge_adapter.py` (rewritten in Phase 20-04) has
only ever been exercised against a `train_attn=True, train_mlp=False` probe LoRA.** It has never merged a
`train_mlp=True` (MoE expert) delta — which is exactly what every real GEN-02/JUDGE-02 adapter will be. The
OLD base's MoE merge convention lived in a separate, base-specific script
(`scripts/merge_tinker_v3.py`, unfused per-expert `gate_proj`/`up_proj`/`down_proj` 3D tensors) that does
not apply to the new base's fused `mlp.experts.{gate_up_proj,down_proj}` `nn.Parameter` tensors
(`config/train_config_v4.yaml` CR-01 comment). Whether Tinker's MoE LoRA export for the new 256-expert
checkpoint uses the same 3-tensor SHARED/PER-EXPERT convention `merge_tinker_v3.py` documented, a different
convention, or is even mergeable via PEFT's `target_parameters` path at all, is **unverified** — this must
be probed with a cheap, real Tinker MoE run (mirroring 20-04's attention-only probe pattern) before Stage
2/3's real adapters are trained, not discovered after a $2-6 SFT run completes.

The second key finding: **`output_router_logits=True`** (STACK.md's new-finding SFT-config item) is sourced
from HF's *raw `transformers` forward()* usage notes — the layer `scripts/train_model.py`/Unsloth uses, NOT
the layer Tinker's `create_lora_training_client()`/`forward_backward()` API operates at. It is **unverified**
whether Tinker's cloud training API exposes an equivalent knob at all; Tinker abstracts the forward pass
away from raw HF kwargs. Do not assume this maps 1:1 onto `tinker_reasoning_sft.py` — verify against
Tinker's actual training-client API surface before treating it as an actionable config line.

**Primary recommendation:** Fork `tinker_reasoning_data.py`/`tinker_reasoning_sft.py` into v4-parameterized
siblings (new `BASE_MODEL`, verify/replace `RENDERER_NAME`), run a cheap real-Tinker MoE-LoRA merge probe
(rank 8, `train_mlp=True`, few steps) to de-risk the merge path BEFORE the real GEN-02/JUDGE-02 runs, do
JUDGE-01's format-compliance smoke on the raw base in parallel (cheap, no training dependency), then run
Stage 2 and Stage 3 in parallel (they share only Phase 20 as a dependency) using Tinker's own LR
(`hp.get_lr`, empirically ~4.99e-4 for rank-32 MoE-only LoRA on the old base per ROADMAP.md's Phase 4.3
supersession note — see Open Questions on the GEN-02 "LR ≤2e-5" text conflict), reuse the Tinker-capture-based
promotion eval (cheap, no merge needed) for iteration, and defer the vLLM-served 8192-token-cap gate
measurement (JUDGE-03's literal requirement text) to the point right before the final merge/packaging
decision — exactly the sequencing v1.3 itself used (Tinker-capture promotion 2026-07-04, vLLM-served
ensemble measurement later at Phase 15 packaging).

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GEN-01 | Thinking-mode/`<think>` SFT data-format decision recorded + rendered-example spot-check (no spurious empty `<think></think>` blocks, QwenLM #131); max tokenized length asserted under Tinker 64K cap | See "GEN-01" below — existing data format is plain single-turn text (NOT native `<think>` blocks), renderer question is whether an equivalent `qwen3_disable_thinking`-style renderer exists for the new base in `tinker_cookbook` |
| GEN-02 | Generation model SFT completes on reasoning mix (reuse Stage-1 data, MoE-only LoRA r32, LR ≤2e-5, frozen router, `output_router_logits=True`) | `scripts/tinker_reasoning_sft.py` is the exact driver to fork; see "GEN-02" below for the LR conflict and the `output_router_logits` API-layer mismatch |
| GEN-03 | Gen model clears wp-bench floor 0.4286 (CI lower bound; or freshly-measured noise-adjusted floor, measured not assumed) | `scripts/bench_wpbench_base_anchor.py` + `scripts/run_eval_reasoning.py::_run_wpbench` is the reusable harness; requires a working merge path first |
| JUDGE-01 | Judge-output-format-compliance smoke on the raw pre-SFT base early (community-reported 18% noncompliance) | `config/judge_system.md` (9-dim rubric) + `eval/output_parsers.py::parse_judge_scores` is the parser to reuse; base-anchor serving pattern from `bench_wpbench_base_anchor.py` is the harness precedent |
| JUDGE-02 | 3-seed relabel-SFT (seeds {1,0,2}) completes reusing v1.3 labels (`data/relabel_v1/`) | `scripts/tinker_reasoning_sft.py --stage full --epochs 3 --seed {1,0,2}` on `data/reasoning_dataset/openai_train_relabel_v1.jsonl` is the exact v1.3 invocation pattern |
| JUDGE-03 | Judge rho measured vs held-out relabeled val (`scripts/relabel/eval_relabel.py`, vLLM-served, 8192-token cap) against pre-registered targets | `scripts/relabel/eval_relabel.py` (Spearman + bootstrap CI, exact v1.3 script) + `scripts/capture_judge_responses_tinker.py` (Tinker-side capture) — see sequencing note on Tinker-capture-first vs vLLM-served-final |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| LoRA SFT training (gen + judge) | Tinker cloud (remote) | — | No `load_state`/weight-continuation; each stage is a fresh LoRA-from-base cloud job (`PIPELINE.md` Prerequisites) |
| Data loading / rendering | Tinker cloud (via `tinker_cookbook`) | Local (data assembly already done, Stage 1 reuse) | `FromConversationFileBuilder` + `renderers.get_renderer()` run inside the Tinker training-client call |
| Checkpoint export / merge | Local (GB10, CPU-heavy) | — | `scripts/merge_adapter.py` downloads the Tinker checkpoint archive and merges onto local base weights |
| wp-bench / judge-rho eval (final, vLLM-served) | Local GPU (GB10, vLLM container) | — | Requires the merged model served over HTTP; `scripts/run_eval_reasoning.py`/`serve_v4_judge_vllm.sh`-style container |
| wp-bench / judge-rho eval (iteration, Tinker-capture) | Tinker cloud (SamplingClient) | — | `scripts/capture_judge_responses_tinker.py` samples directly against the Tinker checkpoint, no merge/serve needed — this is how v1.3's promotion decision was actually made |
| Format-compliance smoke (JUDGE-01) | Local GPU (raw base, vLLM) | Tinker cloud (alternative) | Either the local `serve_base20_vllm.sh` (Phase 20 harness) or a Tinker `SamplingClient` against the raw (LoRA-free) base model works; local vLLM matches the eventual shipping-stack harness more closely |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `tinker` + `tinker_cookbook` | pinned by `.venv-tinker` (version not re-verified this pass — see Open Questions) | Cloud LoRA SFT client, renderers, dataset builders | This project's ENTIRE Stage 2/3 training path since the 2026-06 Tinker pivot; `scripts/tinker_reasoning_sft.py`/`tinker_reasoning_data.py` |
| `transformers` | 5.x, exact min UNPINNED (per Phase 20 research, gated by the bring-up smoke test) [CITED: `.planning/research/STACK.md:154`] | Local model load for merge (`merge_adapter.py`), tokenizer | Required for `Qwen3_5MoeForConditionalGeneration` class resolution |
| `peft` | 0.18.1 (confirmed live, `output/base20/load_smoke.json` per `20-VERIFICATION.md`) | Local `PeftModel`/`get_peft_model` reconstruction inside `merge_adapter.py` | Already verified present and working on this exact host for the attention-only probe merge |
| `vllm` | ≥0.19.0 (confirmed live at `0.20.2rc1...`, `output/base20/deltanet_smoke.json`) | Serving the raw base (JUDGE-01 smoke) and the merged model (GEN-03/JUDGE-03 final eval) | Already proven working with CUDA-graph capture on this exact checkpoint in Phase 20 |
| `scipy` (`spearmanr`) + `numpy` | already in project venv | Judge-rho computation + bootstrap CI | `scripts/relabel/eval_relabel.py` uses these directly, unchanged |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `docker` (via `scripts/dgx_toolbox.py` / `_p0_vllm_smoke_serve.py`) | n/a | Container-based vLLM serving on GB10 | Every local serve step (merge verification, JUDGE-01 smoke, GEN-03/JUDGE-03 final eval) |
| `kernels` (Hub-kernel path) | optional | DeltaNet acceleration on GB10/SM121 (`use_kernels=True`, 1.38x prefill) | Not required for Stage 2/3 training itself (that's remote/Tinker); relevant only if local serving speed for eval matters — Phase 20 decided `use_kernels=False` for the DeltaNet smoke, re-confirm that decision still applies at Phase 21's eval scale |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Tinker-capture-based promotion iteration (`capture_judge_responses_tinker.py`) | Full merge→vLLM-serve→eval for every seed/epoch | Merge+serve is ~66 GiB CPU copy + GPU boot cycle per checkpoint; Tinker capture is a pure API call against the already-existing sampler checkpoint — this is exactly why v1.2/v1.3 used Tinker capture for iteration and reserved merge+serve for the final shipping-figure measurement only |
| Manual LR override (≤2e-5 per GEN-02's literal text) | Tinker's auto `hp.get_lr(BASE_MODEL, is_lora=True)` | The actual v1.2/v1.3 runs used Tinker's auto LR (~4.99e-4 per ROADMAP.md), not a manual ≤2e-5 cap — see Open Questions |

**Installation:** No new packages required — `.venv-tinker` (Tinker SDK) and the project's existing eval venv already have everything Phase 21 needs. `TINKER_API_KEY` is present in `.env` (confirmed by `20-04-SUMMARY.md`, not exported to shell by default).

**Version verification:** `peft==0.18.1`, `transformers==5.3.0`, `vllm==0.20.2rc1.dev196+g84f7a5534.d20260510`, `huggingface_hub==1.23.0` were all live-confirmed on the GB10 host during Phase 20 (`output/base20/load_smoke.json`, `output/base20/deltanet_smoke.json`). No `npm`/`pip` package-legitimacy audit applies — no new external packages are being introduced in this phase; every library above is already installed and proven working on this exact host.

## Package Legitimacy Audit

Not applicable — Phase 21 introduces no new third-party packages. All libraries (`tinker`, `tinker_cookbook`, `transformers`, `peft`, `vllm`, `scipy`, `numpy`) are already installed and were live-verified during Phase 20 (see Standard Stack above). No `npm view`/`pip index versions` audit is needed.

## How v1.2 (gen SFT) + v1.3 (judge relabel-SFT) actually ran — file by file

This is the literal "same pipeline" the user's directive refers to. Every file below is CURRENTLY hardcoded
to the OLD base and needs a v4 fork/parameterization before Phase 21 can run it.

### Data layer

- **`data/reasoning_dataset/`** — the gen-model reasoning mix. `metadata.json`: 704 total examples (563
  train / 141 val, 80/20 split), mix = 60.1% CoT / 24.9% CtF / 15.1% replay (D-05 mix, NOT the
  V4-RERUN-ROADMAP's paraphrase "reasoning + 30% judge replay + 20% wp_gen replay" — that text describes an
  EARLIER Phase 4.2 spec; the actual shipped `metadata.json` mix is 60/25/15 CoT/CtF/replay). Training
  examples are plain single-turn `{"messages": [{"role":"user","content":"<wp_gen>..."}, {"role":"assistant",
  "content":"..."}]}` JSONL rows — verified by direct inspection of `openai_train.jsonl` row 0/1. The
  assistant completion for `<wp_judge>` rows is prose-per-dimension text (`"WPCS Compliance: score 9/10 —
  ..."`), NOT JSON-first — the `<judge_output>{...}</judge_output>` JSON block (parsed by
  `eval/output_parsers.py`) appears later in the same completion, not shown in the first 600 chars sampled.
  No `<think>` tags anywhere in the stored training data — the format predates Qwen3.6 entirely.
- **`data/relabel_v1/`** — the judge relabel-SFT targets. `labels.json` (603 human-relabeled items, M=3
  median aggregation), `gt_dims_train.json`/`gt_dims_val.json` (per-dimension ground truth). Consumed by
  `data/reasoning_dataset/openai_train_relabel_v1.jsonl` (478 judge targets recalibrated from
  `labels.json` per `output/tinker/PROMOTED_v1.3.json`).
- **`data/reasoning_dataset/openai_val.jsonl`** — the held-out val set every eval script (`eval_relabel.py`,
  `student_gap.py`) indexes against; row filtering by `content.startswith("<wp_judge>")` in file order is the
  index-alignment contract `capture_judge_responses_tinker.py`'s docstring calls out explicitly — this
  discipline MUST be preserved for the new base's captures.

### Stage 2 — gen SFT (v1.2)

- **`scripts/tinker_reasoning_data.py`** — data adapter. Hardcodes `BASE_MODEL = "Qwen/Qwen3-30B-A3B"`,
  `RENDERER_NAME = "qwen3_disable_thinking"`, `MAX_LENGTH = 8192`. Uses `tinker_cookbook`'s
  `FromConversationFileBuilder` (built-in OpenAI-chat-format consumer, no custom dataset class) +
  `ChatDatasetBuilderCommonConfig(train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)`. Docstring explains the
  `<wp_gen>`/`<wp_judge>` tokens train as PLAIN-TEXT LITERALS under Tinker's stock tokenizer (Tinker doesn't
  use the project's locally-extended tokenizer) — verified surviving tokenize/decode round-trip.
- **`scripts/tinker_reasoning_sft.py`** — the training driver (same driver for smoke/short/full stages, only
  `--max-steps`/`--epochs` differ). Key mechanics:
  - `sc.create_lora_training_client(base_model=BASE_MODEL, rank=args.rank, train_mlp=True,
    train_attn=args.train_attn [default False], train_unembed=args.train_unembed [default False])` — this IS
    the "MoE-only LoRA" config (`train_mlp=True` always on = routed MoE experts get LoRA; attention/unembed
    off by default per D-N1).
  - LR: `lr = hp.get_lr(BASE_MODEL, is_lora=True)` — Tinker's own auto-computed LR for the base+LoRA
    combination, NOT a manually-set value. `hp.get_lr` takes the base model name as input — **for the new
    base, this will look up (or fail to look up) `Qwen/Qwen3.6-35B-A3B` in whatever internal table
    `tinker_cookbook.hyperparam_utils` uses**, an unverified execution-time dependency (see Open Questions).
  - Training loop: per-batch `tc.forward_backward(data=batch, loss_fn="cross_entropy")` (or the
    `json_weighted` lever, an optional up-weighting of the `<judge_output>` JSON span — a documented but
    NON-default lever, `--loss json_weighted --json-weight 3.0`) + `tc.optim_step(tinker.AdamParams(...))`.
  - Per-epoch: `tc.save_weights_for_sampler(name=..., ttl_seconds=None)` (persistent sampler checkpoint,
    manifest written incrementally EVERY epoch — the durability fix from the 2026-06-07 driver rewrite).
  - Final: `terse_eval()`/`terse_gate_eval()` measure the fraction of val completions LACKING
    `[/REASONING]` (the REVL-05 format-collapse metric) — Wilson-95-upper-bound gated at `rate<=0.10` /
    `wilson_upper<=0.15`, n≥300 via multi-sample-per-prompt at temp>0.
  - `--save-state` additionally calls `tc.save_state(...)` for a durable training-checkpoint (distinct from
    the sampler-only checkpoint) — needed if a future continuation/RL warm-start off this exact run is ever
    wanted (irrelevant to Phase 21 itself, but the flag exists and costs nothing extra to set).
- **Merge:** `scripts/merge_tinker_v3.py` (OLD-base-specific — per-expert `gate_proj`/`up_proj`/`down_proj`
  3D tensors, Tinker's THIRD distinct MoE convention, empirically verified by tensor inspection 2026-06-07)
  and the newer, prefix-aware `scripts/merge_adapter.py` (Phase 20-04, base-agnostic-by-design but ONLY
  tested against `train_attn=True/train_mlp=False`). **Neither script has been proven against a real
  `train_mlp=True` (MoE) Tinker delta on the NEW base** — this is the single largest unverified link in the
  whole rerun (see Common Pitfalls #1).
- **Gate:** wp-bench via `scripts/run_eval_reasoning.py::_run_wpbench` (reused unmodified by
  `scripts/bench_wpbench_base_anchor.py` for a raw-base anchor run on the OLD base — same
  `request_timeout=1800s`, `max_tokens=2048`, `concurrency=4`, `enable_thinking=False`,
  `temperature=0.0`, seed 1337, real-generation warm-up gate before trusting throughput). Known v1.2 result:
  wp-bench 0.4484 vLLM (bar 0.4286).

### Stage 3 — judge relabel-SFT (v1.3)

- **Same driver, different data + seeds + epochs:** `python scripts/tinker_reasoning_sft.py --stage full
  --epochs 3 --seed {1,0,2} --train-path data/reasoning_dataset/openai_train_relabel_v1.jsonl` is the literal
  invocation pattern (`PIPELINE.md` Stage 3 entrypoint line, corroborated by
  `output/tinker/wp-reasoning-relabel-s1-manifest.json`/`-s2-manifest.json` naming). `--rank 32` (default),
  `train_mlp=True`/`train_attn=False`/`train_unembed=False` (same MoE-only convention as gen).
- **Promotion eval (cheap, iteration-speed):** `scripts/capture_judge_responses_tinker.py` samples DIRECTLY
  against the Tinker `SamplingClient` (`sc.create_sampling_client(model_path=tinker_path)`) — **no merge, no
  vLLM, no local GPU needed for this step.** Default `--max-tokens 1024` (⚠ too low per carry-forward lesson
  1 — a 2026-07-09 observation recorded "Fresh s1 judge captures systematically truncated; 119/121 responses
  lack clean endings" at this default; the 8192-token cap discipline from PKG-03's later fix must be applied
  to the Tinker-capture path too, not just the final vLLM-served measurement). Writes `{index, response}`
  JSONL with an index-alignment contract that MUST match `eval_judge._run_eval_reasoning`'s row filtering.
- **Scoring:** `scripts/relabel/eval_relabel.py` — reads the capture JSONL, parses via
  `eval.output_parsers.parse_judge_scores(text, "auto")`, computes Spearman rho vs
  `output/relabel/val_labels_v1.json`, bootstrap CI (2000 resamples, seed 7), prints delta vs the
  `student_gap.json` baseline and the `sqrt(rel_M3)` attenuation ceiling. This is the EXACT script GEN-03/
  JUDGE-03's requirement text names — it needs zero code changes for the new base (it only reads capture
  JSONL + label JSON, both base-agnostic file formats); only the CAPTURE step upstream of it changes.
- **Promotion record:** `output/tinker/PROMOTED_v1.3.json` — the single-seed s1 winner
  (`judge_rho_vs_new_val_labels: 0.8274`, n=121, measured via the Tinker-capture path above, NOT vLLM-served)
  was promoted with `"local_export": "...; merge deferred to Phase 11 packaging"` — **the actual v1.3
  promotion decision used the cheap Tinker-capture number, and the vLLM-served 0.8075 ensemble figure the
  roadmap quotes as "the number to beat" was only produced later, after merging, during Phase 15 packaging's
  Q8 ensemble measurement** (`output/packaging/pkg03_ens8192_results.json`). This is the sequencing pattern
  Phase 21 should mirror: cheap Tinker-capture iteration first, vLLM-served final-figure measurement last.
- **Ensemble:** 3-seed median ensemble computed downstream (packaging-phase concern, not Phase 21's — but
  Phase 21 must produce and PRESERVE all 3 seed checkpoints/manifests so a later phase can ensemble them,
  same as `output/tinker/wp-reasoning-relabel-s1-manifest.json`/`-s2-manifest.json` plus the s1-promoted-as-
  v1.3 primary).

## GEN-01: thinking-mode/`<think>` SFT data-format decision

**Finding:** The existing training data (`data/reasoning_dataset/openai_train.jsonl`) contains **zero**
native `<think>` tags — completions are plain prose (`"WPCS Compliance: score 9/10 — ..."`) followed later
by a `<judge_output>{...}</judge_output>` JSON block for judge rows [VERIFIED: direct file read]. The OLD
base's renderer, `qwen3_disable_thinking`, was chosen specifically because "our format is IN-BAND prose +
`[/REASONING]` + `<judge_output>` JSON, NOT native `<think>` blocks, so we do not want the thinking renderer
injecting `<think>` scaffolding" (`tinker_reasoning_data.py` docstring). This means Pitfall 6's concern
(spurious empty `<think></think>` blocks corrupting the loss target) is likely **N/A for this project's
specific format** — IF an equivalent no-thinking-scaffolding renderer exists for the new base in
`tinker_cookbook`.

**The actual open question:** does `tinker_cookbook.renderers` ship a `Qwen3.6`/`Qwen3.5`-family renderer at
all, and if so, is there an equivalent to `qwen3_disable_thinking` (Qwen3.6 uses `enable_thinking:false` in
`chat_template_kwargs`, an API/serving-config setting per `FEATURES.md` #4, not the old model's dynamic
`/think`/`/nothink` prompt toggle the `qwen3_disable_thinking` renderer name implies) [ASSUMED — renderer
availability for this specific new-base architecture was not verified this research pass; `tinker_cookbook`'s
renderer registry was not inspected directly]. **Recommend:** as a Wave-0 task, list
`tinker_cookbook.renderers`'s available renderer names/check for a `qwen3_5`/`qwen3.6`-specific entry before
committing to a renderer name in the forked `tinker_reasoning_data.py`; if none exists, the fallback is
constructing the generation/training prompt manually (bypass `apply_chat_template()` entirely, matching how
the existing data already bypasses native `<think>` formatting) — this sidesteps issue #131 entirely rather
than working around it.

**Rendered-example spot-check + 64K-cap assert (also GEN-01):** trivial — the same pattern
`tinker_reasoning_data.py.__main__` already runs (`b[0].model_input.length`) should be extended to print
`max(len)` across the full train+val set and assert `< 64_000` (Pitfall 8's Tinker training-context cap);
given the old base's examples were function-level PHP + rubric prose comfortably inside 8192
(`MAX_LENGTH = 8192` in the existing adapter), clearing 64K is expected with large margin — measure, don't
assume, per the roadmap's own discipline.

## GEN-02: generation model SFT completes

**MoE-only LoRA r32, frozen router:** directly satisfied by forking `tinker_reasoning_sft.py` unchanged
(`--rank 32` default, `train_mlp=True`/`train_attn=False`/`train_unembed=False` default = MoE-only,
router untouched because Tinker's `create_lora_training_client` never exposes a `train_router` flag in this
driver — same "router frozen by omission" discipline the RL phase used, D-09-02).

**LR ≤2e-5 — literal requirement conflicts with actual pipeline behavior.** `tinker_reasoning_sft.py` uses
`lr = hp.get_lr(BASE_MODEL, is_lora=True)` (Tinker's own auto-computed LR), not a manual value. ROADMAP.md's
Phase 4.3 goal note states explicitly: "RTRN-01/02/03 ... are local-DGX-framed and SUPERSEDED by the Tinker
regime (**LR 4.99e-4**, cloud LoRA)" [VERIFIED: `.planning/ROADMAP.md` Phase 4.3 goal text] — i.e. the actual
LR the real v1.2/v1.3 runs used was ~4.99e-4, roughly **25x higher** than GEN-02's literal "≤2e-5" text. The
≤2e-5 figure is a carry-over from the abandoned DGX/Unsloth-era `RTRN-01` spec (superseded 2026-06-11) that
appears to have been copy-pasted forward into the new v4.0 GEN-02 requirement text without re-deriving it
against the Tinker regime that actually shipped v1.2/v1.3. **This is flagged as an Open Question below —
the planner must resolve it explicitly, not silently pick one.**

**`output_router_logits=True` — API-layer mismatch, unverified for Tinker.** STACK.md's finding is sourced
from HF's `transformers` usage notes for the architecture (a raw `model(...)` forward-pass kwarg) [CITED:
huggingface.co/docs/transformers/model_doc/qwen3_5_moe]. This is the exact kwarg `scripts/train_model.py`
(local DGX/Unsloth path, currently UNUSED for Stage 2/3 — Tinker is the actual training venue per
`PIPELINE.md` Prerequisites) would pass to `AutoModelForCausalLM(...)`. A repo history note found during this
research shows the OLD base's local Unsloth path *disabled* `output_router_logits` due to an Unsloth
incompatibility (May 2026) — the **opposite** direction from STACK.md's new-base recommendation, underscoring
that this is a real, non-trivial per-stack decision, not a copy-paste default. Whether Tinker's cloud
`create_lora_training_client`/`forward_backward` API exposes an equivalent knob, defaults it on/off
internally, or the concept doesn't apply at Tinker's abstraction layer at all, is **unverified** — the
Tinker Python SDK source / docs were not inspected for this specific API surface this research pass.
**Recommend:** an execution-time check of `tinker`'s training-client method signatures / Tinker docs for any
MoE load-balancing/router-aux-loss control before assuming this line item is actionable inside
`tinker_reasoning_sft.py`; if Tinker doesn't expose it, the finding may simply be inapplicable to this
project's actual training venue and should be recorded as such rather than forced in.

## GEN-03: wp-bench floor 0.4286

Directly reuses `scripts/run_eval_reasoning.py::_run_wpbench` + the `boot_vllm`/`wait_healthy`/`generate`
harness from `scripts/_p0_vllm_smoke_serve.py`, exactly as `scripts/bench_wpbench_base_anchor.py` already
demonstrates for a raw-foundation-model anchor run (real-generation warm-up gate before trusting throughput
— the Phase 15 LOCKED lesson, carry-forward lesson 2). **Blocking dependency: this requires a WORKING merge
of the real GEN-02 adapter** — see Common Pitfalls #1. Bar 0.4286 is the carried-forward v3.0 acceptance
floor (`PIPELINE.md`/`V4-RERUN-ROADMAP.md` Stage 2(c)); V4-RERUN-ROADMAP explicitly allows "a freshly-derived
floor if the new base's baseline coding ability materially shifts the noise band" — the raw base-anchor
number from the OLD base's own recent measurement (Qwen3-30B-A3B raw: 0.4033 overall,
`output/bench17/wpbench_base_anchor.json`, committed 2026-07-12) is the template for how to produce an
equivalent NEW-base raw anchor if the CI-lower-bound gate needs a freshly-measured comparison point instead
of the inherited 0.4286.

## JUDGE-01: judge-output-format-compliance smoke on raw base

**What "judge output format" concretely means:** the `<judge_output>{...}</judge_output>` JSON block with
9-dimension scores (`config/judge_system.md`'s rubric: WPCS/SQL/Security/Performance/API-usage/Code-
quality/Dependency/i18n/Accessibility), parsed by `eval/output_parsers.py::parse_judge_scores(text, "auto")`
→ `_parse_json_scores`/`_parse_prose_scores`. `strip_think()` (a pre-existing `<think>...</think>` regex
strip, already generic — not Qwen3.6-specific, will work unchanged on the new base's always-on thinking
blocks) runs before any parse attempt [VERIFIED: `eval/output_parsers.py:78-80`].

**How to smoke it:** reuse the exact `bench_wpbench_base_anchor.py` pattern — serve the RAW (no adapter)
`Qwen/Qwen3.6-35B-A3B` via `scripts/serve_base20_vllm.sh` (Phase 20-03's v4 serve script, already proven
working with `--language-model-only` / `LANGUAGE_MODEL_ONLY` env gating), real-generation warm-up gate, then
feed N (~20-50, cheap) `<wp_judge>`-prefixed prompts drawn from `data/reasoning_dataset/openai_val.jsonl`
(the base has no task-token training yet, so `<wp_judge>` will be consumed as plain text — same posture the
old-base raw anchor took toward `<wp_gen>`), run `parse_judge_scores` against the raw completions, and
measure the parse-fail rate. Compare against the community-reported **18% output-format-noncompliance** rate
[CITED: `FEATURES.md` #2b, an informal 2026-05-16 community gist, not vendor-verified] — this is exactly the
failure mode that "killed 3/4 ratios on the old base" per the roadmap's own framing (SUMMARY.md). Do this
EARLY and in parallel with Stage 2 gen SFT — it has zero dependency on GEN-02/JUDGE-02 completing and is
cheap (no training, single serve + N generations).

## JUDGE-02: 3-seed relabel-SFT (seeds {1,0,2})

Directly reuses `tinker_reasoning_sft.py --stage full --epochs 3 --seed {1,0,2}` against
`data/reasoning_dataset/openai_train_relabel_v1.jsonl` — unchanged data, unchanged seed set, unchanged
epoch count. The only delta is `BASE_MODEL`/`RENDERER_NAME` (same fork as GEN-02) and the LR/`output_router_
logits` open questions above apply identically to the judge track. **Re-open condition (already resolved by
V4-RERUN-ROADMAP discretion item 2, not Phase 21's decision to make):** reuse the v1.3 labels as-is; only
reopen the human relabel campaign if the new base's SFT saturates below 0.85/0.87 AND a gap-closure-style
diagnostic (mirroring `output/relabel/gap_closure_summary.json`'s 2026-07-08 investigation pattern) rules out
training-recipe causes.

## JUDGE-03: judge rho measured against pre-registered targets

**Sequencing recommendation (see "How v1.3 actually ran" above):** run the cheap Tinker-capture eval
(`capture_judge_responses_tinker.py` → `eval_relabel.py`) per seed/epoch for FAST iteration — but with
**`--max-tokens` raised to 8192**, not the script's current default of 1024, per carry-forward lesson 1
(truncation looked like a quality regression on the old base until the cap was raised; a 2026-07-09
observation already caught 119/121 truncated captures at the 1024/lower default during v1.3's own
development). Only after a seed clears the pre-registered floor on the cheap path should the expensive
merge→vLLM-serve→8192-cap measurement run — this is the literal JUDGE-03 requirement text ("vLLM-served,
8192-token cap") and it is also what actually produced the v3.0 shipping figure (0.8075 ensemble), NOT what
produced the promotion decision (0.8274 Tinker-capture single-seed). **Both numbers are real, valid, and
different measurement methodologies — the plan must be explicit about which one gates what**, mirroring the
distinction the old base's own history shows.

**Targets:** rho **>0.85 single-seed OR >0.87 3-seed median ensemble**, vs the 0.8075/0.8017 v3.0 shipping
wall (`V4-RERUN-ROADMAP.md` pre-registered success criteria). Failure disposition: recorded as a valid,
measured outcome per the roadmap's explicit "no_winner is a result" discipline — not a phase failure to
force past.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (project-wide convention; `tests/test_download_model_v4.py`, `tests/test_check_token_alignment.py` from Phase 20 are the closest precedent for this milestone) |
| Config file | none dedicated — project root `pytest.ini`/`pyproject.toml` conventions apply (not inspected this pass; Phase 20 tests ran via plain `pytest tests/test_X.py -x -q`) |
| Quick run command | `pytest tests/test_tinker_reasoning_data_v4.py -x -q` (new file, to be created in Wave 0) |
| Full suite command | `pytest tests/ -k "phase21 or v4" -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GEN-01 | max tokenized example length < 64K; renderer resolves without error for the new base | unit | `pytest tests/test_tinker_reasoning_data_v4.py::test_max_length_under_cap -x` | ❌ Wave 0 |
| GEN-02 | Tinker training client accepts `train_mlp=True` MoE-only config for the new base; loss decreases over first N steps (smoke stage) | integration (real Tinker spend, cheap) | `python scripts/tinker_reasoning_sft_v4.py --stage smoke --max-steps 4` (manual, not pytest — matches existing `--stage smoke` convention) | ❌ Wave 0/1 (forked script) |
| GEN-03 | merged model serves + wp-bench score ≥ floor (CI lower) | integration (real GPU) | `python scripts/run_eval_reasoning.py` (wp-bench path) against the merged v4 gen model | ✅ harness exists, needs merge first |
| JUDGE-01 | raw-base judge-format parse-fail rate measured and recorded | integration (real GPU, cheap) | new smoke script, mirrors `bench_wpbench_base_anchor.py` | ❌ Wave 0 (new script) |
| JUDGE-02 | 3 seeds complete training without divergence; per-seed manifest + sampler checkpoint persisted | integration (real Tinker spend) | `python scripts/tinker_reasoning_sft_v4.py --stage full --epochs 3 --seed {1,0,2}` (manual) | ❌ Wave 0/1 (forked script) |
| JUDGE-03 | judge rho (Tinker-capture) computed per seed; final vLLM-served 8192-cap rho computed for promoted seed(s) | integration | `python scripts/capture_judge_responses_tinker.py ...` → `python scripts/relabel/eval_relabel.py ...` (both reused unmodified) | ✅ scripts exist, need v4 base-param + merge for the final step |
| Merge-path MoE probe (blocking prerequisite, not a REQ-ID) | a real `train_mlp=True` Tinker LoRA merges cleanly onto the new base via `merge_adapter.py` (or its extension) | integration (real Tinker spend, cheap, mirrors 20-04's probe) | new `scripts/build_base20_moe_probe_adapter.py` (mirrors `build_base20_probe_adapter.py` but `train_mlp=True`) | ❌ Wave 0 (new script) — **highest-priority Wave 0 item** |

### Sampling Rate

- **Per task commit:** run the relevant smoke/dry-run command for whatever was just changed (data adapter
  test, or a cheap `--stage smoke --max-steps 4` Tinker call) — do not wait for a full epoch to catch a
  config error.
- **Per wave merge:** re-run `pytest tests/test_download_model_v4.py tests/test_check_token_alignment.py -x`
  (Phase 20's tests — confirms nothing in the shared config/download/merge machinery regressed) plus any new
  Phase 21 test files.
- **Phase gate:** GEN-03 wp-bench full run + JUDGE-03 vLLM-served ensemble measurement green before
  `/gsd-verify-work`.

### Wave 0 Gaps

- [ ] `scripts/build_base20_moe_probe_adapter.py` (or equivalent) — a REAL, cheap Tinker `train_mlp=True`
  LoRA probe run against the new base, proving the MoE merge path before any real GEN-02/JUDGE-02 spend.
  This is the single highest-value Wave 0 item this research identifies — mirrors 20-04's exact successful
  pattern (rank=8, few steps, ~cents) but for `train_mlp=True` instead of `train_attn=True`.
- [ ] `scripts/tinker_reasoning_data_v4.py` / parameterize `tinker_reasoning_data.py` for
  `BASE_MODEL="Qwen/Qwen3.6-35B-A3B"` + resolve the `RENDERER_NAME` open question (GEN-01).
  `tests/test_tinker_reasoning_data_v4.py` — data-adapter build succeeds, batch count sane, max length < 64K.
- [ ] `scripts/tinker_reasoning_sft_v4.py` (or a `--base-model` CLI flag on the existing driver) — forked/
  parameterized training driver.
- [ ] New JUDGE-01 raw-base format-compliance smoke script (mirrors `bench_wpbench_base_anchor.py`
  structurally, but runs the judge-format parser instead of wp-bench).
- [ ] Confirm `merge_adapter.py`'s WR-02/WR-03/WR-04 review-fix commits (post-dating the passing
  `vl_merge_roundtrip.json` receipt per `20-VERIFICATION.md` carry-forward item 2) are re-exercised by a
  fresh `python scripts/smoke_vl_merge_base20.py` run before Phase 21's real adapters depend on the guard
  logic — inherited carry-forward, not a new item, but must not be silently skipped.

## Common Pitfalls

### Pitfall 1: MoE-expert (`train_mlp=True`) merge path is completely unproven on the new base

**What goes wrong:** every real GEN-02/JUDGE-02 adapter uses `train_mlp=True` (MoE-only LoRA, the whole
point of the recipe). `merge_adapter.py` — the ONLY merge script proven against this base at all — was
tested exclusively with `train_attn=True, train_mlp=False` (an attention-only probe, deliberately scoped
that way per `20-04-SUMMARY.md`'s own key-decision log: "Tinker's MoE per-expert convention is a distinct,
already-solved problem (`scripts/merge_tinker_v3.py`) out of scope for this merge-path smoke"). But
`merge_tinker_v3.py` is hardcoded to the OLD base's UNFUSED per-expert 3D tensor convention
(`gate_proj`/`up_proj`/`down_proj` separate 2D-per-expert matrices) — the new base's experts are FUSED
(`mlp.experts.gate_up_proj`/`down_proj` raw `nn.Parameter` tensors, requiring PEFT's `target_parameters`
mechanism per `config/train_config_v4.yaml`'s own CR-01 comment). Neither script is known to handle this.

**Why it happens:** Phase 20 deliberately scoped the merge-path smoke to attention-only to keep the probe
cheap and isolated; the MoE convention was explicitly deferred as "already-solved" based on the OLD base's
solution, without re-verifying that solution transfers.

**How to avoid:** run a cheap real Tinker probe with `train_mlp=True` (rank 8, few steps, mirrors
`build_base20_probe_adapter.py`'s exact successful pattern) BEFORE the real GEN-02/JUDGE-02 spend, download
the checkpoint archive, and inspect the actual exported tensor keys/shapes against the new base's fused
`mlp.experts.*` parameters — exactly the empirical-first discipline 20-04 itself demonstrated for the
attention/DeltaNet case (which found a real, non-obvious mismatch: DeltaNet's split `in_proj_q/k/v` vs the
checkpoint's fused `in_proj_qkv`). Do not assume the MoE case will be cleaner just because it's a smaller
scope than DeltaNet.

**Warning signs:** merge guard's `merged_target_module_count != expected_target_module_count`; or (worse)
a merge that "succeeds" with all-zero deltas if `target_parameters` silently doesn't wire the fused tensors.

**Phase to address:** Phase 21, Wave 0, before any real Stage 2/3 training spend.

### Pitfall 2: `output_router_logits` recommendation may not be actionable at Tinker's abstraction layer

See "GEN-02" above. Do not add this as an unqualified config line in the forked training driver without
first confirming Tinker's training-client API surface actually exposes an equivalent control — the
STACK.md finding is sourced from raw-`transformers`-forward()-call usage notes, a different layer than
Tinker's cloud API operates at, and a repo-history note shows the OLD base's local Unsloth path went the
OPPOSITE direction (disabled it) for compatibility reasons.

### Pitfall 3: GEN-02's "LR ≤2e-5" text vs the actual Tinker-regime LR (~4.99e-4) — do not silently pick one

See "GEN-02" above and Open Questions below. The requirement text is very likely a stale carry-over from
the superseded DGX/Unsloth-era spec; picking the wrong one silently either (a) trains at an LR 25x lower
than what actually worked before (risking under-convergence within budget) or (b) trains at an LR the
literal requirement text forbids (risking a spurious "requirement violated" verification finding even
though it's what the real pipeline does). Resolve explicitly, record the decision and rationale.

### Pitfall 4: Tinker-capture eval default `--max-tokens 1024` silently truncates judge captures

Already empirically observed on the old base (2026-07-09: "119/121 responses lack clean endings" at low
token caps) and independently confirmed by the PKG-03 packaging-phase fix (raising 2048→8192 fixed an
apparent quality regression that was actually truncation). `capture_judge_responses_tinker.py`'s CLI default
is `--max-tokens 1024` — lower than even the already-too-low 2048 that caused the packaging-phase scare.
Always pass `--max-tokens 8192` explicitly for any real JUDGE-02/JUDGE-03 capture.

### Pitfall 5: naming collision — "v4" means two different things in this repo

`scripts/serve_v4_judge_vllm.sh` and `models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4` refer to the
**v4-winner reasoning-adapter grid** from Phase 04.3/04.4 (an OLD-base artifact, "v4" = 4th grid iteration),
completely unrelated to the "v4.0" MILESTONE (Qwen3.6 rerun) this phase belongs to. Do not reuse or branch
from these scripts assuming they're new-base-related; they serve the OLD base's RL-era judge and have
nothing to do with Qwen3.6.

### Pitfall 6: chat-template empty `<think>` blocks — likely N/A, but must be explicitly verified, not assumed

See GEN-01 above. The project's existing data format bypasses native `<think>` rendering entirely (plain
prose + `[/REASONING]` + `<judge_output>` JSON), so QwenLM issue #131 (empty `<think></think>` in historical
turns) likely doesn't apply — but this must be confirmed against whatever renderer the new base actually
uses, not carried over unverified, per PITFALLS.md's own explicit caveat on this exact point.

## Open Questions

1. **GEN-02's "LR ≤2e-5" vs the actual Tinker-regime ~4.99e-4 — which governs the new-base run?**
   - What we know: the literal REQUIREMENTS.md/ROADMAP.md text for GEN-02 says "LR ≤2e-5"; the actual
     v1.2/v1.3 runs used Tinker's auto-computed LR (`hp.get_lr`), empirically ~4.99e-4, and ROADMAP.md's own
     Phase 4.3 text says the ≤2e-5-era requirements (RTRN-01/02/03) were EXPLICITLY SUPERSEDED by the Tinker
     regime.
   - What's unclear: whether GEN-02's v4.0 text is a deliberate NEW constraint (tighter LR for the new,
     presumably more capable base) or an uncorrected copy-paste of the old, superseded DGX-era number.
   - Recommendation: default to Tinker's auto `hp.get_lr()` value (matching "follow the same pipeline"
     literally) and record the resolved numeric LR in the plan/verification artifact; if the ≤2e-5 text was
     intentional, surface this conflict to the user/discuss-phase rather than silently overriding a written
     requirement.

2. **Does `tinker_cookbook` expose a renderer for `Qwen/Qwen3.6-35B-A3B` (or its `Qwen3.5MoE` family), and
   is there a no-thinking-scaffolding equivalent to `qwen3_disable_thinking`?**
   - What we know: the old base used `RENDERER_NAME = "qwen3_disable_thinking"`; the new base's thinking
     mode is controlled via `chat_template_kwargs.enable_thinking` (an API/serving-config setting per
     `FEATURES.md`), architecturally different from the old base's prompt-toggle.
   - What's unclear: whether `tinker_cookbook.renderers` ships an entry for this exact new architecture at
     all (not inspected this pass).
   - Recommendation: Wave-0 task, inspect `tinker_cookbook.renderers`'s registry directly before writing the
     forked data adapter; fall back to manual prompt construction (bypassing `apply_chat_template()`
     entirely) if no suitable renderer exists — this is lower-risk than debugging an unfamiliar renderer's
     `<think>`-block behavior mid-training.

3. **Does Tinker's training-client API expose an `output_router_logits`-equivalent control, or is
   STACK.md's finding inapplicable at this abstraction layer?**
   - What we know: the finding is sourced from raw-`transformers` usage notes for direct `model(...)` calls;
     Tinker abstracts the forward pass.
   - What's unclear: whether Tinker's `create_lora_training_client`/`forward_backward` exposes any MoE
     load-balancing/router-aux-loss knob, or handles it internally by default (in which case there is
     nothing for Phase 21 to configure).
   - Recommendation: check the Tinker Python SDK's training-client method signatures / official Tinker docs
     for MoE-specific training controls before treating this as an actionable Stage 2/3 config line; record
     the finding either way (applicable-and-set, or confirmed-inapplicable) rather than leaving it silently
     unaddressed.

4. **Is the new base's Tinker MoE LoRA export using the same SHARED/PER-EXPERT 3-tensor convention
   `merge_tinker_v3.py` documented for the old base, or a different convention (given the fused
   `mlp.experts.gate_up_proj`/`down_proj` parameter layout)?**
   - What we know: the old base's experts were unfused 2D per-expert tensors targeted via PEFT
     `target_modules`; the new base's experts are fused raw `nn.Parameter` tensors requiring PEFT's
     `target_parameters` mechanism (per `config/train_config_v4.yaml`'s CR-01 comment) — a structurally
     different merge problem.
   - What's unclear: how Tinker's own LoRA-export format represents a `target_parameters`-style fused-expert
     delta, and whether `merge_adapter.py`'s prefix-aware per-key resolution logic (built for standard
     `nn.Linear` `target_modules`) can handle it at all without extension.
   - Recommendation: this is exactly what Pitfall #1's Wave-0 MoE probe is for — do not proceed to the real
     GEN-02/JUDGE-02 spend until this is empirically answered.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Tinker API key / cloud training | GEN-02, JUDGE-02, all Tinker spend | ✓ | `.env` `TINKER_API_KEY`, confirmed present 2026-07-13 (`20-04-SUMMARY.md`) | none — hard dependency, no local-GPU fallback exists for training this model size |
| GB10 local GPU (vLLM serving) | GEN-03, JUDGE-01, JUDGE-03 (final vLLM-served step) | ✓ | proven working Phase 20 (`deltanet_smoke.json`, `vl_merge_roundtrip.json`) | none needed, already confirmed |
| `transformers`/`peft`/`vllm` versions | merge + serve | ✓ | 5.3.0 / 0.18.1 / 0.20.2rc1 (Phase 20 receipts) | none needed |
| Disk space for merged checkpoints (~65-67 GiB each) | GEN-03/JUDGE-03 merge step | ✓ | 2.5 TB free confirmed (`20-04-SUMMARY.md`) | none needed |

**Missing dependencies with no fallback:** none identified — Phase 20 already proved every piece of local
infrastructure this phase needs (download, load, serve, merge scaffold). The only genuinely unverified
items are the Tinker-API-surface and renderer-availability questions above (Open Questions 1-4), which are
execution-time verification tasks, not missing tooling.

## Security Domain

Not applicable in the ASVS sense — this phase performs no new user-facing input handling, authentication, or
session management. The relevant "security" concern is entirely covered by the existing Security dimension
(dimension 3) of `config/judge_system.md`'s rubric, which is DATA the judge model is trained to score, not
infrastructure this phase must itself secure. `TINKER_API_KEY` handling (present in `.env`, not exported to
shell by default, sourced explicitly under `.venv-tinker`) follows the existing project convention and needs
no new work.

## Sources

### Primary (HIGH confidence — direct repo file reads this session)

- `.planning/ROADMAP.md` (Phase 4.3 supersession note — the LR 4.99e-4 finding; Phase 20 section; v4.0
  milestone structure)
- `.planning/REQUIREMENTS.md` (GEN-01..03, JUDGE-01..03 exact text; v4.0 traceability table)
- `.planning/V4-RERUN-ROADMAP.md` (Stage 2/3 stage map, pre-registered success criteria, discretion items)
- `.planning/phases/20-base-bring-up/20-VERIFICATION.md` (BASE-01..04 receipts, carry-forward items 1-2)
- `.planning/phases/20-base-bring-up/20-04-SUMMARY.md` (Tinker MoE-vs-attention merge-scope decision, the
  DeltaNet in_proj mismatch precedent, tokenizer-vocab-mismatch fallback, disk/API-key state)
- `config/train_config_v4.yaml` (CR-01 fixed LoRA target_modules/target_parameters)
- `PIPELINE.md` (Stage 2/3 entrypoints, gates, known v1.2/v1.3 results)
- `scripts/tinker_reasoning_sft.py`, `scripts/tinker_reasoning_data.py`, `scripts/relabel/eval_relabel.py`,
  `scripts/capture_judge_responses_tinker.py`, `scripts/merge_adapter.py` (headers/key functions),
  `scripts/merge_tinker_v3.py` (header/convention docstring), `scripts/train_model.py` (apply_lora),
  `scripts/bench_wpbench_base_anchor.py`, `scripts/serve_v4_judge_vllm.sh`, `eval/output_parsers.py`,
  `config/judge_system.md`, `data/reasoning_dataset/metadata.json`, `data/reasoning_dataset/openai_train.jsonl`
  (direct sample read), `output/tinker/PROMOTED_v1.3.json`

### Secondary (MEDIUM confidence — synthesized research from a prior session, re-verified 2026-07-12)

- `.planning/research/SUMMARY.md`, `STACK.md`, `PITFALLS.md`, `FEATURES.md` (output_router_logits finding,
  chat-template bug #131, Tinker 64K cap, 18% judge-format-noncompliance community report, Tinker pricing)

### Tertiary (LOW confidence — flagged explicitly in-line)

- Renderer availability for the new base in `tinker_cookbook` (Open Question 2) — not verified this pass.
- Whether Tinker's training-client API exposes an `output_router_logits`-equivalent (Open Question 3) — not
  verified this pass.
- Whether the new base's Tinker MoE LoRA export uses the same 3-tensor convention as the old base (Open
  Question 4) — not verified this pass, flagged as the top Wave-0 priority.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every library/version was live-confirmed on the actual host during Phase 20, days before this research.
- Architecture (how the old pipeline ran): HIGH — read every relevant script directly, cross-checked against actual receipts/manifests, not just docs.
- New-base deltas (MoE merge convention, renderer availability, Tinker LR/router-logits API surface): MEDIUM-LOW — these are the genuine unknowns this research surfaces as Wave-0 verification items, not assumptions to build a plan on top of.
- Pitfalls: HIGH — every pitfall cites either a direct file read or a specific dated repo observation, not speculation.

**Research date:** 2026-07-13
**Valid until:** ~7 days (fast-moving milestone; any Tinker API/pricing change after 2026-07-17 price rise, or any Wave-0 probe result, should trigger a re-read of this document's Open Questions before planning proceeds).
