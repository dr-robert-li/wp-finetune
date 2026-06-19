# Phase 9: GSPO Training - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-20
**Phase:** 9-gspo-training
**Areas discussed:** Algorithm + platform, Gen vs judge budget, Judge reward definition, Router-shift policy

---

## Algorithm + platform

### Q1 — Execution platform

| Option | Description | Selected |
|--------|-------------|----------|
| Tinker (cloud) | Same venue as SFT; only place 30B fits; custom RL loop on Tinker primitives; router-telemetry risk | ✓ |
| Research-gated | Don't lock; researcher confirms Tinker RL + any DGX path first | |
| DGX-local (GB10) | ROADMAP's assumption; full router access; but bf16 30B can't load locally | |

**User's choice:** Tinker (cloud) — LOCKED (D-09-01)
**Notes:** GB10 cannot host 30B bf16 (proven). ROADMAP Phase 9 DGX-execution text is stale — flagged for correction.

### Q2 — Router-training scope (frozen vs trained gates)

| Option | Description | Selected |
|--------|-------------|----------|
| Keep router frozen | LoRA on experts/attn/shared (SFT-matching); shift ~0; mask monitor-only; relaxes GRPO-06 router clause | |
| Train router gates | Honor GRPO-06 full-MoE; needs Tinker router-gate targeting + telemetry; unproven | |
| Research-gated | Researcher confirms Tinker LoRA module-targeting + routing telemetry; lock at plan time | ✓ |

**User's choice:** Research-gated (D-09-02)
**Notes:** Frozen-vs-trained determines whether router-shift + protected-expert regularizer are load-bearing or monitor-only.

### Q3 — GSPO vs GRPO

| Option | Description | Selected |
|--------|-------------|----------|
| GSPO primary, GRPO fallback-only | GSPO per D-08; GRPO only if GSPO infeasible; no planned comparison | |
| GSPO + GRPO comparison | Run both; ~2x compute; Phase 10 already does eval | |
| Research-gated | Confirm GSPO-loop feasibility on Tinker primitives first | ✓ |

**User's choice:** Research-gated (D-09-03) — GSPO remains primary per D-08 unless infeasible on Tinker
**Notes:** GSPO needs sequence logprobs (Tinker provides); RSPO floor feasibility is the open item.

---

## Gen vs judge budget

| Option | Description | Selected |
|--------|-------------|----------|
| Interleaved, judge-weighted | Single run, each batch mixes gen+judge, judge ≥ 50%; both pathways guard regression | ✓ |
| Judge-heavy, light gen | Most budget on judge; gen anti-regression touch-up only | |
| Two sequential stages | Gen then judge (or vice versa); clean attribution but catastrophic-forgetting risk | |

**User's choice:** Interleaved, judge-weighted (D-09-04)
**Notes:** ~60/40 judge/gen start. Gen's verifiable reward is the anti-regression anchor (gen-regression seen once before).

---

## Judge reward definition

| Option | Description | Selected |
|--------|-------------|----------|
| Claude-agent internal-consistency rubric | Spawned Claude Code agent rates score-reasoning consistency 0–1; reuse Phase 8 dispatch; + fix-correctness | ✓ (revised) |
| Agreement-with-gold | Score vs held-out gold rubric; circularity risk; that's Phase 10's metric | |
| Research-gated | Researcher surveys consistency-reward formulations | |

**User's choice:** Claude-agent rubric — REVISED after web research (D-09-05)
**Notes:** User asked whether a Claude agent introduces training regression. Research confirmed KNOWN failure modes (reward hacking, self-preference bias, reward-noise → GRPO advantage collapse). Decision kept BUT hardened with mandatory guards (fix-correctness anchor, capped consistency reward, temp=0 + N-vote, anti-hack regression gate) and three research-gated items: (R1) Claude→Qwen self-preference on Claude-distilled CoT lineage, (R2) reward-noise budget vs GRPO SNR, (R3) Claude-agent latency/non-stationarity in the rollout→gradient loop.

---

## Router-shift policy

| Option | Description | Selected |
|--------|-------------|----------|
| Two-tier (auto-recover, then halt) | Protected-expert deactivation → auto-inject regularizer + retry epoch; hard shift breach → halt + human | ✓ |
| Halt + human on any breach | Any trip stops training; conservative; interrupts on transient noise | |
| Auto-recover only | Always regularize + retry; hard-halt only after N failed retries | |

**User's choice:** Two-tier (D-09-06)
**Notes:** CI-aware thresholds (D-09). Active only if router trained (D-09-02); monitor-only if frozen.

---

## Claude's Discretion

- Exact judge/gen interleave ratio (60/40 start; constraint judge ≥ gen).
- Concrete numeric thresholds: router-shift, protected-expert frequency, consistency-reward cap, N-vote count.

## Deferred Ideas

- Router-RL deferred if D-09-02 resolves frozen (router reshaped in Phase 13 anyway).
- GSPO vs GRPO empirical comparison → Phase 10 eval.
- Noise-corrected/Dr.GRPO adoption → only if R2 shows collapse-regime reward noise.
- Reviewed-not-folded todos: `phase8-inherit-judge-recalibration.md`, `phase7-8-ci-aware-noiseband-gates.md` already satisfied by Phase 8 — recommend clearing.
