# Phase 09 — POST-V6 Handover (self-contained continuation runbook)

**Written:** 2026-06-26 · **Branch:** `phase10-execution`
**Predecessor:** `09-V2-V6-HANDOVER.md` (the V2–V6 runbook) is DONE — executed in full; see
`09-V2-V6-RESULTS.md` for the acceptance report. **This doc is what comes NEXT**, runnable from a cleared
context with zero prior conversation: the cheap pre-spend probe, then the 500-step go/no-go (Dr. Li's gate).

---

## 0. TL;DR — where things stand

V2–V6 verification of the reward fixes is COMPLETE:
- **V2 PASS** (echo-adversary 0.25 ≤0.30; no saturation; gap compressed).
- **V3 PASS** — but only after finding+fixing a NEW live bug (below).
- **V4 PASS** — 15-step RL smoke, all gates; `frac_groups_all_zero` 0.375→0.0 live.
- **V5 PASS (with caveat)** — echo 0.25 on the trained policy; no inflation; manual block-read clean.
- **V6a PASS** (additive-guard); **V6b DEFERRED to RLEV-01** (false-green on a 15-step smoke).

**NEW bug found + fixed this round (committed `8cf35c1`):** the chat EOS marker `<|im_end|>` leaked into the
DECODED gen completion text (`tok.decode` had no `skip_special_tokens`), so bare-code gen completions failed
`php -l` → gen reward zeroed for the entire gen axis. A confirmed (likely dominant) cause of the dead gen
gradient. Fix = `scripts/rl_rollouts.py:1220` `tok.decode(tokens, skip_special_tokens=True)` (TEXT only;
`.tokens`/`.logprobs` keep the EOS token for the GSPO IS ratio). Suite 118→120 green.

**NOT a 500-step green light.** The smoke shows WEAK judge-axis learning: the V4 policy emits parseable
same-function code in ~12.5% of judge completions vs warm-start's 30%, and NEVER improves (hi-tier=0). The
open question is whether the judge-axis signal is strong enough to justify 500 steps. **Do the §5 truncation
probe FIRST** — it may be a cheap fix for the weak signal.

---

## 1. Current state (commits on `phase10-execution`)

```
f0e8079  docs(phase-09): V2-V6 verification complete — results + diagnostic probes
8cf35c1  fix(phase-09 reward): strip chat EOS marker from decoded gen completions (dead-gradient cause)
264a772  docs(phase-09): exhaustive V2-V6 verification handover (Tinker spend approved)
318855e  fix(phase-09 reward): correctness pressure (judge) + template-aware validity (gen)
ff0872e  fix(phase-09 RL): restore KL autohalt guard + fix stale-sampler (non-learning) bug
```
All reward fixes (FIX 1 correctness pressure, FIX 2 template validity, the decode strip, KL/sampler fixes)
are LIVE in `scripts/rl_rollouts.py` `collect_rollouts` — any fresh RL run uses them automatically.
Reward-code tree is clean (the only `git status` modifications are unrelated data/planning artifacts).

**Test baseline (must stay green) — 120 pass:**
```bash
.venv-tinker/bin/python -m pytest tests/test_reward_fix1_judge.py tests/test_reward_fix2_gen.py \
  tests/test_rl_train.py tests/test_rl_train_integration.py tests/test_reward_pipeline.py \
  tests/test_rl_rollouts_reward_shape.py -q -k "not test_lora_config"
```

---

## 2. Environment setup (run before ANY Tinker / RL step)

```bash
cd /home/robert_li/Desktop/projects/wp-finetune
set -a; . ./.env; set +a                 # loads TINKER_API_KEY
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN   # MANDATORY: keep the $0 local-judge path off paid API
PY=.venv-tinker/bin/python
```

**Local judges ($0, required):** must be UP.
```bash
curl -s localhost:8000/v1/models | grep -o wp_judge          # -> wp_judge
curl -s localhost:8001/v1/models | grep -o wp_consistency    # -> wp_consistency
# If down: bash scripts/serve_v4_judge_vllm.sh ; GPU_MEM_UTIL=0.22 bash scripts/serve_consistency_vllm.sh
```

**OOM guard — MANDATORY before any RL run.** DGX Spark (GB10) has NO OOM protection; an OOM hangs the host
unrecoverably. Arm exactly ONE (launch in an isolated command — combining kill+launch trips exit 144):
```bash
setsid bash -c 'bash scripts/_oom_guard.sh > logs/oom_guard.$(date +%s).log 2>&1' </dev/null & disown
ps -eo pid,args | grep '[_]oom_guard.sh' | grep -vE 'snapshot-bash|eval '   # confirm exactly one guard loop
```
NEVER run more than two 30B vLLM models at once. (As of this writing one guard IS already armed and both
judges ARE up — verify before relying on it.)

---

## 3. Key artifacts (paths you will need)

**Tinker checkpoints (sampler paths):**
- WARM-START (RL init, v4 SFT savestate):
  `tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state`
- V4 step-15 (NEW-reward 15-step smoke; the policy V5 evaluated):
  `tinker://83dd8be5-1b4c-5337-9330-526db3f20ad3:train:0/sampler_weights/step-15`  (also `.../final-step-15`)
- STALE step-50 (negative control): `tinker://a99724f2-36d3-577b-b51f-94af9198e7d8:train:0/sampler_weights/step-50`

**Harnesses (committed):**
- `scripts/_check_judge_fixcorr.py` — controlled JUDGE-axis eval (NEW reward) + echo-adversary + PAIR-CHECK.
  This is the step-50 gate instrument for the 500-step run.
- `scripts/_dump_corrected_blocks.py` — prints `_extract_corrected_php` blocks per policy (manual read).
- `scripts/_v3_liveness.py` — robust per-prompt-TIMEOUT gen-axis liveness probe (use instead of
  `_check_step50_vs_warmstart.py`, which hangs on a single pathological prompt's blocking `sample().result()`).
- `scripts/_v3_dump.py` — raw gen-completion + `php -l` dump (the tool that found the `<|im_end|>` leak).
- `scripts/_launch_post81_rerun.sh <seedNum> <suffix>` — RL launcher (hardcodes --total-steps 500,
  --checkpoint-every 50).

**Reward fns (all in `scripts/rl_rollouts.py`):** `_fix_score_from_completion(completion, original_code)`
(FIX 1, tiers 0/0.25/0.5/→1.0), `_judge_original_code(item)`, `_is_valid_wp_php(code)` (FIX 2),
`_generate_completions` (line ~1220 holds the decode-strip fix). Gen path uses `_is_valid_wp_php`
(rl_rollouts.py:957); judge path passes `_judge_original_code` into `_fix_score_from_completion` (:993-994).

**Metrics/logs:** `output/rl_checkpoints/metrics/` (`rl_metrics.<suffix>.jsonl`, `manifest.<suffix>.json`),
`logs/phase09_rerun/`.

**Reference:** `09-V2-V6-RESULTS.md` (full V2–V6 results + the recommendation), `09-V2-V6-HANDOVER.md`
(the executed runbook), `09-REWARD-FIX-DESIGN.md`, `09-LOCAL-RL-STATUS-UPDATES.md`.

---

## 4. Reward tier table (so you can judge outputs)

`_fix_score_from_completion(completion, original_code)`:
| input | score |
|---|---|
| no corrected block | 0.0 |
| non-empty, unparseable PHP (incl. pure critique prose) | 0.25 |
| parseable, identity-FAIL (diff fn name OR <60% token retention) | 0.25 |
| parseable, identity-OK, no rubric improvement (reproduction) | 0.50 |
| parseable, identity-OK, rubric improves Δ → | `0.5 + 0.5*min(1, Δ/0.30)` up to 1.0 |

`_is_valid_wp_php`: plain `php -l` pass → True; template markers (`<#`,`<%`,`{{`) + a php toggle →
neutralize directives + lint; broken template/standalone PHP → False. Marker-bearing strings → False (the
EOS strip is at DECODE, not in the gate).

---

## 5. NEXT STEPS — run in order

### STEP A — `judge_max_new_tokens` truncation probe (CHEAP, do this FIRST)

**Why:** V5 showed the V4 policy emits a corrected ```php block in only ~12.5% of judge completions. The SFT
format is critique-THEN-fix; the verbose per-dimension critique may exhaust `judge_max_new_tokens` BEFORE
the corrected-code block is emitted (the truncation failure documented at `rl_rollouts.py:1132`). If so, the
"weak judge-axis signal" is a budget artifact, not policy weakness — and raising the budget is a cheap lever
worth far more than re-benching the smoke.

**Find the current budget:** `grep -n "JUDGE_MAX_NEW_TOKENS" scripts/rl_rollouts.py` (and how
`_check_judge_fixcorr` / `_dump_corrected_blocks` pass `judge_max_new_tokens`).

**Probe (dump V4 + warm judge completions at a LARGER budget, compare code-emission rate):**
```bash
# _dump_corrected_blocks uses JUDGE_MAX_NEW_TOKENS internally; add a --judge-max-new-tokens override
# (mirror _generate_completions' max_tokens_override) OR temporarily raise JUDGE_MAX_NEW_TOKENS, then:
V4=tinker://83dd8be5-1b4c-5337-9330-526db3f20ad3:train:0/sampler_weights/step-15
$PY -m scripts._dump_corrected_blocks --fixed-50 "$V4" --n-prompts 12 \
  > logs/phase09_rerun/A_trunc_probe.log 2>&1
# count code-emission rate at the larger budget:
python3 - <<'EOF'
import re; t=open('logs/phase09_rerun/A_trunc_probe.log').read()
b=re.split(r'################ (\S+) ################', t)
for i in range(1,len(b),2):
    print(b[i], "raw'<?php'=", b[i+1].count('<?php'), "prompts=", b[i+1].count('prompt#'))
EOF
```
**Decision:**
- Code-emission rate RISES materially at a larger budget → truncation was the cause. Bump the live
  `judge_max_new_tokens` for the 500-step run; the weak-signal concern is largely resolved. Proceed to STEP B
  with more confidence.
- No change → the policy genuinely under-emits fixes. Revisit judge-axis reward shaping / advantage strength
  (handover NULL-branch lead) BEFORE spending. Surface to Dr. Li; do not auto-proceed to 500 steps.

### STEP B — 500-step RL re-spend (ONLY on explicit Dr. Li authorization)

This is a **separate human gate**, not implied by V2–V6 passing. If authorized:
```bash
set -a; . ./.env; set +a; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
# OOM guard armed (exactly one), judges up. Two seeds:
bash scripts/_launch_post81_rerun.sh 42 seedA     # --total-steps 500 --checkpoint-every 50
bash scripts/_launch_post81_rerun.sh 7  seedB
```
**Gate at step 50 on the JUDGE-axis controlled eval — NOT the live slope** (the live per-step slope is too
noisy; J.6). Run the V2 instrument on each seed's step-50 checkpoint:
```bash
$PY -m scripts._check_judge_fixcorr --fixed-50 "<seedA step-50 path>" \
  --n-prompts 20 --group-size 2 --temperature 0.2
# PASS signal: echo-adversary ≤0.30 AND fixed-50 meaningfully > warm-start with stale ≈ warm
# (the negative control is load-bearing; a fixed win with stale ALSO > warm is a confounded eval).
```
Then RLEV-01 (full wp-bench codegen regression vs v1.2 SFT — incl. the DEFERRED V6b) + RLEV-02 on the
completed policy.

---

## 6. Decision gates (summary)

| gate | where | PASS condition |
|---|---|---|
| Reward hack-resistant | V2/V5 (DONE) | echo-adversary ≤0.30 ✓ |
| Gen gradient live | V4 (DONE) | `frac_groups_all_zero` <0.375 ✓ (0.0) |
| Reward non-regressing (mech.) | V6a (DONE) | broken PHP still rejected ✓ |
| Judge-axis signal strong enough | **STEP A (open)** | code-emission rate acceptable after budget probe |
| 500-step authorization | **STEP B (Dr. Li)** | human gate |
| Codegen no-regression (empirical) | RLEV-01 | V_final wp-bench ≥ v1.2 SFT within noise |

---

## 7. Caveats (carry forward)

1. **Read the DATA, not the score (bit us 3× + once more this round).** The gen `parseable=0` looked like a
   FIX-2 failure until raw completions were dumped → it was the `<|im_end|>` decode leak. Always dump raw
   completions (`_v3_dump.py` / `_dump_corrected_blocks.py`) before concluding.
2. **Rubric under-credits real fixes (limiter).** `_extract_verifiable_signals` needs a fuller
   nonce+`wp_unslash`+`sanitize_text_field`+`esc_html` fix to move a dimension; `esc_html` alone didn't.
   Some genuine fixes are under-credited → the hi-tier (>0.5) may stay empty even when the policy improves.
   If STEP A shows fixes scoring low, consider scoring Δ on the SPECIFICALLY FLAGGED dimension, not overall.
3. **`_check_step50_vs_warmstart.py` hangs** on a single pathological prompt (blocking `sample().result()`,
   no timeout). Use `_v3_liveness.py` (per-prompt timeout) for gen-axis sampling instead.
4. **Each RL launch is a fresh warm-start (NO resume)** from the v1.2 savestate at step 0.
5. **`.env` carries ANTHROPIC_API_KEY** — always `unset` it (per §2) so the $0 local-judge path can't leak.
6. **Don't fight false-greens.** A wp-bench / regression number on a ≤15-step smoke is meaningless (the LoRA
   barely moved). Empirical codegen regression belongs to RLEV-01 on a fully-trained policy.

---

## 8. Quick reference — one-liners

```bash
# regression suite (expect 120 pass)
.venv-tinker/bin/python -m pytest tests/test_reward_fix1_judge.py tests/test_reward_fix2_gen.py tests/test_rl_train.py tests/test_rl_train_integration.py tests/test_reward_pipeline.py tests/test_rl_rollouts_reward_shape.py -q -k "not test_lora_config"
# offline reward hack-probe (no Tinker): echo 0.25 / gut 0.25 / repro 0.5 / fix ~0.59
.venv-tinker/bin/python - <<'EOF'
from scripts.rl_rollouts import _fix_score_from_completion
o="<?php\nfunction render($x){ $q=$_GET['q']; echo $q; }"
f=lambda p:"x\n```php\n"+p+"\n```"
for n,p in [("echo","<?php echo 'hi';"),("gut","<?php function render($x){ return 1; }"),("repro",o),("fix","<?php function render($x){ $q=sanitize_text_field(wp_unslash($_GET['q'])); echo esc_html($q); }")]:
    print(n, round(_fix_score_from_completion(f(p),o),3))
EOF
# confirm the decode fix is live (gen completions must be marker-free)
grep -n "skip_special_tokens" scripts/rl_rollouts.py     # -> line ~1220
# stop an RL run
kill $(cat output/rl_checkpoints/metrics/rl_run.<suffix>.pid)
```

---

## 9. LOCKED GATE PLAN — live RL rerun (Dr. Li, 2026-06-27)

**Decision (LOCKED).** Run the controlled judge-axis eval (`_check_judge_fixcorr.py` — fixed
held-out prompts + low temp + deterministic reward; NOT the noisy live `fix_correctness_mean`
slope) at EVERY step-50 checkpoint: 50, 100, 150, 200, 250, … through 500.

- **Binding flat-gate at step 250.** If `fixed-N` is STILL flat vs warm-start (`fixed ≈ warm`,
  no clear improvement) by **step 250** → **STOP** the run (sampler necessary-but-not-sufficient →
  gradient-strength `adv × lr` issue; revisit before re-spending). Bounded exposure ≤250 steps
  (~50% of run), within the approved 500-step spend.
- **Otherwise** (a clear `fixed > warm` shows up at any read ≤250) → **push to 500**, continuing
  the every-50 reads (300/350/400/450/500) as monitoring.
- **Early-kill (any checkpoint, overrides the above).** Clear breakage: KL/efrac auto-halt,
  `reward_mean` collapse, regression (`fixed << warm`), or hack (echo-adversary > 0.30).
- **Per-read PASS shape:** `fixed-N` meaningfully > warm-start **AND** `stale-ctrl ≈ warm-start`
  (the negative control is load-bearing — a fixed win with stale ALSO > warm = confounded, means
  nothing).
- **Seeds:** seedA runs first to the gate; **seedB launches only on a clear pass** (avoids
  saturating the shared local judges :8000/:8001 with two concurrent runs).
- **Per-checkpoint eval cmd:**
  `_check_judge_fixcorr --fixed-50 <stepN ckpt sampler_path> --n-prompts 20 --group-size 2 --temperature 0.2`
- **Run config:** `judge_max_new_tokens=4096` wired in (commit 9ff2a94); warm-start = v4
  r32-rp30 MoE-only savestate; `_launch_post81_rerun.sh 42 seedA`.
