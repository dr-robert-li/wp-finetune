# Phase 09 — V2–V6 Verification Handover (exhaustive, self-contained)

**Written:** 2026-06-26 · **Branch:** `phase10-execution` · **Tinker spend for V2–V6: APPROVED by Dr. Li.**
**Goal:** execute the reward-fix verification suite V2–V6 (`09-REWARD-FIX-DESIGN.md`) end-to-end from a
cleared context. V1 (unit + hack-probe) is already DONE and green. This doc is everything needed to run
V2–V6 without prior conversation context.

> **STATUS — EXECUTED 2026-06-26 (this runbook is DONE).** V2 PASS · V3 PASS · V4 PASS · V5 PASS (caveat) ·
> V6a PASS · V6b deferred to RLEV-01. Full results: **`09-V2-V6-RESULTS.md`**. What to run next (pre-spend
> truncation probe → 500-step Dr. Li gate): **`09-POST-V6-HANDOVER.md`**.
> **One material addendum to this runbook (read before any RE-RUN):** executing §5 V3 surfaced a THIRD
> reward defect beyond the two below — the chat EOS marker `<|im_end|>` leaked into the DECODED gen text
> (`tok.decode` lacked `skip_special_tokens`), so every bare-code gen completion failed `php -l` → gen
> reward zeroed. FIX 2 alone could not fix this. Fixed (`scripts/rl_rollouts.py:1220`,
> `tok.decode(tokens, skip_special_tokens=True)`, committed `8cf35c1`); validated LIVE in V4 —
> `frac_groups_all_zero` 0.375→0.0. See the §5 V3 "EXECUTION ADDENDUM" below and the regression baseline
> is now **120 pass** (was 118).

---

## 0. TL;DR — what you are verifying and why

Two reward defects were found during the post-8.1 RL rerun and FIXED (committed):
1. **Judge reward had no correctness pressure** — `_fix_score_from_completion` scored the corrected code
   IN ISOLATION → any clean parseable PHP ~1.0 (trivial echo == real fix). FIX 1 adds an identity gate
   (same function + ≥60% token retention) + improvement-delta vs the original.
2. **Gen reward killed valid template code** — Elementor `content_template()` bodies failed `php -l` →
   scalar 0 → dead gradient. FIX 2 (`_is_valid_wp_php`) neutralizes template directives + lints.

Also fixed earlier (separate, keep): KL autohalt guard (side-channel) + stale-sampler (per-step refresh).

V2–V6 prove the fixes work LIVE and do not regress, BEFORE authorizing a full 500-step RL re-spend.
**Do NOT launch a full 500-step run from this doc** — V4 is a ≤15-step smoke; the 500-step go/no-go is a
separate Dr. Li gate AFTER V5/V6 pass.

---

## 1. Current state (commits on `phase10-execution`)

```
318855e  fix(phase-09 reward): correctness pressure (judge) + template-aware validity (gen)   <- FIX 1 + FIX 2 + tests + design doc
ba24475  docs(phase-09): ticket — wp_gen reward path contributes zero gradient
543e806  diag(phase-09 RL): controlled judge-axis eval + block dumper; reward-design flaw found
ff0872e  fix(phase-09 RL): restore KL autohalt guard + fix stale-sampler (non-learning) bug
```
All reward fixes are LIVE in `scripts/rl_rollouts.py` `collect_rollouts` — any fresh RL run uses the new
reward automatically. No uncommitted code. `git status` should be clean for these files.

**Test baseline (must stay green):**
```
.venv-tinker/bin/python -m pytest tests/test_reward_fix1_judge.py tests/test_reward_fix2_gen.py \
  tests/test_rl_train.py tests/test_rl_train_integration.py tests/test_reward_pipeline.py \
  tests/test_rl_rollouts_reward_shape.py -q -k "not test_lora_config"
# Expect: all pass. (test_lora_config is a pre-existing, unrelated TINKER/MagicMock failure — ignore.)
```

---

## 2. Environment setup (run before ANY Tinker / RL step)

```bash
cd /home/robert_li/Desktop/projects/wp-finetune
set -a; . ./.env; set +a                 # loads TINKER_API_KEY
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN   # MANDATORY: keep $0 local-judge path off paid API
PY=.venv-tinker/bin/python               # the tinker venv interpreter
```

**Local judges ($0, required for gen reward scoring + RL):** must be UP.
```bash
curl -s localhost:8000/v1/models | grep -o wp_judge          # -> wp_judge
curl -s localhost:8001/v1/models | grep -o wp_consistency    # -> wp_consistency
# If down:
bash scripts/serve_v4_judge_vllm.sh                          # wp_judge :8000
GPU_MEM_UTIL=0.22 bash scripts/serve_consistency_vllm.sh     # wp_consistency :8001
```

**OOM guard — MANDATORY before any RL run (V4).** DGX Spark (GB10) has NO OOM protection; an OOM hangs
the host unrecoverably. Arm exactly ONE:
```bash
nohup bash scripts/_oom_guard.sh > logs/oom_guard.$(date +%s).log 2>&1 &
ps -eo pid,args | grep '[_]oom_guard.sh'   # confirm exactly one
```
NEVER run more than two 30B vLLM models at once. Keep the guard armed for the whole of V4.

---

## 3. Key artifacts (paths you will need)

**Tinker checkpoints (sampler paths):**
- WARM-START (RL init, v4 SFT savestate):
  `tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state`
- STALE step-50 (J.4 frozen-sampler run; OLD reward; the V2 NEGATIVE CONTROL):
  `tinker://a99724f2-36d3-577b-b51f-94af9198e7d8:train:0/sampler_weights/step-50`
- FIXED-sampler step-50 (current run; OLD reward; sampler-fix only):
  `tinker://3207bc66-b26a-549a-ad7a-a02cb401eac1:train:0/sampler_weights/step-50`
- (V4 will produce a NEW step-50 trained on the NEW reward — grep its path from the V4 log, §V4.)

**Harnesses (committed; some need the small edits in §V2/§V3 below):**
- `scripts/_check_judge_fixcorr.py`  — controlled judge-axis fix_correctness eval, 3 policies, neg control.
- `scripts/_check_step50_vs_warmstart.py` — gen-axis eval (parseable PHP rate + reward).
- `scripts/_dump_corrected_blocks.py` — prints actual `_extract_corrected_php` blocks per policy.
- `scripts/_probe_weights_moved.py`  — compute_logprobs weight-movement probe + raw-completion dump.
- `scripts/_launch_post81_rerun.sh <seedNum> <suffix>` — RL launcher (env hygiene + full flag set).

**Reward functions (all in `scripts/rl_rollouts.py`):**
- `_fix_score_from_completion(completion_text, original_code=None)` — FIX 1 (tiers: 0.0 / 0.25 / 0.5 / →1.0).
- `_judge_original_code(item)` — extract the original ```php block from a judge prompt item.
- `_primary_php_function_name(code)`, `_token_retention(orig, corr)` — identity-gate helpers (FIX 1).
- `_is_valid_wp_php(code)`, `_neutralize_template_directives(code)` — FIX 2.
- `_extract_verifiable_signals(code)` (in `scripts/reward_pipeline.py`) → `RubricScore.overall` (0–100).

**Metrics / logs:**
- RL metrics dir: `output/rl_checkpoints/metrics/` (per-seed: `rl_metrics.<suffix>.jsonl`, `manifest.<suffix>.json`, `rl_run.<suffix>.pid`).
- RL logs: `logs/phase09_rerun/full_run.<suffix>.log`.
- Eval logs: `logs/phase09_rerun/*.log`.

**Reference docs:** `09-REWARD-FIX-DESIGN.md` (V1–V6 spec), `09-LOCAL-RL-STATUS-UPDATES.md` (J.1–J.8 full
trail), `09-TICKET-gen-reward-dead-gradient.md` (FIX 2 ticket).

---

## 4. The reward fixes in one screen (so you can judge outputs)

`_fix_score_from_completion(completion, original_code)` tiers:
| input | score |
|---|---|
| no corrected block | 0.0 |
| non-empty, unparseable PHP | 0.25 |
| parseable, `original=None` (legacy/back-compat) | `rubric.overall/100` |
| parseable, identity-FAIL (diff function name OR <60% token retention) | 0.25 |
| parseable, identity-OK, no rubric improvement (reproduction) | 0.50 |
| parseable, identity-OK, rubric improves Δ → | `0.5 + 0.5*min(1, Δ/0.30)` up to 1.0 |

Verified offline (V1): trivial echo / unrelated / gutted-short → 0.25; reproduction → 0.5; genuine fix → ~0.59.

`_is_valid_wp_php(code)`: plain `php -l` pass → True; else if template markers (`<#`,`<%`,`{{`) + a php
toggle → prepend `<?php` if bare, neutralize directives, lint. Broken template PHP still → False.

---

## 5. EXECUTION — V2 → V6 (run in order; stop on a FAIL and follow the branch)

### V2 — Controlled judge-axis eval under the NEW reward + hack-resistance (sampling, ~$ cheap)

**Purpose:** prove LIVE that (a) the new reward discriminates (no saturation at 1.0), and (b) an echo
adversary scores ~0.25. Re-scores real policy completions under the NEW (correctness-pressured) reward.

**Required harness edit** — `scripts/_check_judge_fixcorr.py` currently calls
`_fix_score_from_completion(c.completion)` with NO original (legacy path). Change `_score()` to pass the
original and to also report the identity-gate outcome. Apply:

```python
# in _score(), replace the score line. completions come group_size-per-prompt IN ORDER,
# so completion i belongs to prompts[i // args.group_size].
from scripts.rl_rollouts import _fix_score_from_completion, _judge_original_code
gsz = int(getattr(args, "group_size", 2))
scores = []
for i, c in enumerate(comps):
    orig = _judge_original_code(prompts[i // gsz])   # prompts must be in scope in _score (pass it in)
    scores.append(_fix_score_from_completion(c.completion, orig))
```
Pass `prompts` into `_score(name, sampler, prompts, args, renderer, tok)` (it already receives them).
Add a synthetic ECHO-ADVERSARY check at the end of `main()` (no sampling needed — deterministic):
```python
from scripts.rl_rollouts import _fix_score_from_completion, _judge_original_code
echo = "analysis...\n```php\n<?php echo 'hi';\n```"
adv = [_fix_score_from_completion(echo, _judge_original_code(p)) for p in prompts]
print(f"ECHO-ADVERSARY mean_fixcorr = {sum(adv)/len(adv):.4f}  (gate: <= 0.30)")
```

**Run:**
```bash
PY=.venv-tinker/bin/python
$PY -m scripts._check_judge_fixcorr \
  --fixed-50 "tinker://3207bc66-b26a-549a-ad7a-a02cb401eac1:train:0/sampler_weights/step-50" \
  --n-prompts 20 --group-size 2 --temperature 0.2 \
  > logs/phase09_rerun/V2_judge_neweward.log 2>&1
grep -E "mean_fixcorr|ECHO-ADVERSARY|VERDICT" logs/phase09_rerun/V2_judge_neweward.log
```

**GATES (V2 PASS requires ALL):**
- ECHO-ADVERSARY mean ≤ 0.30 (the isolation hack is closed live). **Hard.**
- No policy saturates: under the new reward the per-policy `mean_fixcorr` should be a SPREAD across tiers,
  not pinned near 1.0 (compare to the OLD-reward run in `09-LOCAL-RL-STATUS-UPDATES.md` J.7 where
  warm=0.437, fixed=0.606, stale=0.656 under the LEGACY isolation reward). Under the new reward, expect
  reproductions to land near 0.5 and the warm-vs-step50 gap to SHRINK (the artifact removed).
- Negative-control sanity: stale-step50 should NOT exceed warm-start by a large margin under the NEW
  reward (the J.7 confound was the isolation reward; the new reward should compress it).

**Branches:**
- ECHO > 0.30 → the identity gate is not firing on sampled completions → STOP, inspect
  `_judge_original_code` extraction on real judge prompts (the original ```php block must be found);
  re-check `_dump_corrected_blocks.py` output. Do not proceed.
- All gates pass → V2 PASS, continue to V3.

---

### V3 — Gen-axis liveness under FIX 2 (sampling, ~$ cheap)

**Purpose:** prove the Elementor-template completions that scored 0/60 now earn non-zero, and that a fresh
gen batch is no longer uniformly all-zero.

**Required harness edit** — `scripts/_check_step50_vs_warmstart.py` `_score_policy()` zeroes via
`_is_parseable_php`. Swap to the template-aware guard:
```python
from scripts.rl_rollouts import _is_valid_wp_php          # add import
# in _score_policy: replace `if not _is_parseable_php(p):` with:
if not _is_valid_wp_php(p):
    results[i].scalar = 0.0
else:
    parseable += 1
```

**Run:**
```bash
$PY -m scripts._check_step50_vs_warmstart --n-prompts 15 --group-size 4 \
  > logs/phase09_rerun/V3_gen_liveness.log 2>&1
grep -E "mean_reward|parseable|VERDICT" logs/phase09_rerun/V3_gen_liveness.log
```

**GATES (V3 PASS):**
- warm-start gen `parseable` (now via `_is_valid_wp_php`) > 0 (was 0/60). At least the Elementor-template
  completions are credited. **Hard.**
- At least one policy shows non-zero gen `mean_reward` (was 0.0000 for all). **Hard.**

**Branches:**
- Still 0 parseable → `_is_valid_wp_php` not catching the real completions → dump a few raw gen completions
  (`_probe_weights_moved.py` PROBE 2 already shows them) and widen `_neutralize_template_directives` /
  the bare-body prepend. Re-run V1 `tests/test_reward_fix2_gen.py` after any change.
- Pass → V3 PASS, continue to V4.

> **EXECUTION ADDENDUM (2026-06-26 — what actually happened, fold into any re-run):**
> 1. The committed harness `_check_step50_vs_warmstart.py` **hangs HARD** (`STAT=Sl`, ~0 CPU, wchan
>    `poll_schedule_timeout`) — `_generate_completions` loops per-prompt with a blocking `sample().result()`
>    and no timeout, so one pathological prompt freezes the whole run. Use **`scripts/_v3_liveness.py`**
>    (robust per-prompt timeout, skips hangs) instead of the §V3 harness edit.
> 2. First liveness run = **0/24 parseable by BOTH gates** → the V3-fail branch above (template-widening)
>    was the WRONG lead. Per caveat #2, dumping raw completions (`scripts/_v3_dump.py`) showed every gen
>    completion ends with literal `<|im_end|>`; `php -l` errors "unexpected token '<'". Root cause = the
>    decode special-token leak, NOT templates. **Fix = `rl_rollouts.py:1220` `skip_special_tokens=True`**
>    (TEXT only; `.tokens`/`.logprobs` keep the EOS token for the GSPO IS ratio). Grep first for any
>    `.completion` consumer that keys on the marker (none found — single `tok.decode` site).
> 3. Re-run after fix: **parseable 0→16/24, gen mean_reward 0→0.0416 → V3 PASS.** FIX 2's template credit
>    did not fire in that draw (no Elementor templates sampled) — the live win was the decode fix.
> 4. Added 2 regression tests + a faithful `_FakeTokenizer` double (leaks the marker unless
>    `skip_special_tokens`) so V1 catches this next time. Suite 118→120.

---

### V4 — Short RL smoke on the NEW reward (Tinker spend; ≤15 steps; killable)

**Purpose:** a real RL run with BOTH reward fixes + the committed sampler/KL fixes, to confirm the new
reward produces a healthy live signal and the J.4 `all_zero=0.375` pin is gone.

**Prereqs:** §2 env + judges UP + **OOM guard armed** (mandatory). One seed only.

**Launch (explicit command — the launcher hardcodes 500 steps, so run the flags directly with 15):**
```bash
set -a; . ./.env; set +a; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
MDIR=output/rl_checkpoints/metrics; mkdir -p "$MDIR" logs/phase09_rerun
export WP_JUDGE_DEBUG_DUMP="$MDIR/judge_failures.V4.jsonl"
nohup .venv-tinker/bin/python scripts/rl_train.py \
  --init-from "tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state" \
  --model-id Qwen/Qwen3-30B-A3B --lora-rank 32 --lora-seed 42 \
  --total-steps 15 --batch-size 8 --checkpoint-every 50 --jaccard-every 20 \
  --kl-soft 0.1 --kl-hard 0.3 --efrac-soft 0.7 --efrac-hard 0.5 \
  --judge-base-url http://localhost:8000/v1 --judge-model wp_judge \
  --consistency-base-url http://localhost:8001/v1 --consistency-model wp_consistency \
  --metrics-path "$MDIR/rl_metrics.V4.jsonl" --manifest-path "$MDIR/manifest.V4.json" \
  > logs/phase09_rerun/full_run.V4.log 2>&1 &
echo $! > "$MDIR/rl_run.V4.pid"
```
NOTE: `--checkpoint-every 50` means NO checkpoint at 15 steps. For V5 you need a checkpoint — either set
`--total-steps 50` (≈8h, gets the step-50 ckpt) OR set `--checkpoint-every 15 --total-steps 15`. **Recommend
`--checkpoint-every 15 --total-steps 15`** so V4 produces a usable checkpoint cheaply for V5.

**Monitor (each step ~8–10 min):**
```bash
MDIR=output/rl_checkpoints/metrics
python3 -c "
import json
for r in [json.loads(l) for l in open('$MDIR/rl_metrics.V4.jsonl') if l.strip()]:
    gt9=r.get('frac_reward_gt_0.9');lt1=r.get('frac_reward_lt_0.1')
    mid=round(1-(gt9 or 0)-(lt1 or 0),3) if gt9 is not None else None
    print(f\"s{r['step']:>2} rmean={round(r.get('reward_mean',0),3)} fix={round(r.get('fix_correctness_mean') or 0,3)} \
all_zero={round(r.get('frac_groups_all_zero') or 0,3)} kl={round(r.get('kl_sample_train_v1') or 0,4)} \
ent={round(r.get('entropy') or 0,3)} mid={mid}\")"
grep -E 'refreshed on-policy sampler|HALT|side-channel EMPTY' logs/phase09_rerun/full_run.V4.log | tail
```

**GATES (V4 PASS):**
- Plumbing: `refreshed on-policy sampler -> <id>` logs each step with a CHANGING id (sampler fix live).
- `entropy` ≠ 0.0 and `kl_v1` populated (KL/side-channel fix live — regression check on ff0872e).
- **`frac_groups_all_zero` < 0.375** — the J.4 pin is gone (gen now contributes via FIX 2). **Hard.**
- `fix_correctness_mean` distribution is NOT saturated at ~1.0 every step (FIX 1 discriminating). Expect a
  spread with reproductions ~0.5.
- NO HARD halt, no `side-channel EMPTY` error.

**Branches:**
- `all_zero` still 0.375 → FIX 2 not active in the live gen path → confirm `collect_rollouts` uses
  `_is_valid_wp_php` (grep) and judges are up. STOP + fix.
- fix saturates at 1.0 → FIX 1 not plumbed live → confirm `collect_rollouts` passes
  `_judge_original_code(...)` into `_fix_score_from_completion`. STOP + fix.
- HARD halt → read the halt reason; KL/efrac thresholds. Likely a real divergence — investigate before any
  scale-up.
- All pass → V4 PASS. **Stop the run** (`kill $(cat $MDIR/rl_run.V4.pid)`), keep the checkpoint path
  (grep `Checkpoint saved` in the log).

---

### V5 — Anti-hack acceptance gate (RLEV-02), on the V4 checkpoint

**Purpose:** with a policy trained on the NEW reward, confirm the reward is hack-resistant.

**Run V2's harness on the V4 step-N checkpoint** (substitute the V4 checkpoint path for `--fixed-50`),
plus the echo-adversary already wired in V2:
```bash
V4CKPT=$(grep -oE 'tinker://[^ ]+sampler_weights/step-[0-9]+' logs/phase09_rerun/full_run.V4.log | tail -1)
echo "V4 checkpoint: $V4CKPT"
$PY -m scripts._check_judge_fixcorr --fixed-50 "$V4CKPT" \
  --n-prompts 20 --group-size 2 --temperature 0.2 \
  > logs/phase09_rerun/V5_antihack.log 2>&1
grep -E "mean_fixcorr|ECHO-ADVERSARY|VERDICT" logs/phase09_rerun/V5_antihack.log
```
ALSO re-dump the V4 policy's actual corrected blocks and read them (this is the load-bearing manual check
— do NOT skip; the function-level verification has been wrong twice before when we trusted scores over
reading data):
```bash
$PY -m scripts._dump_corrected_blocks --fixed-50 "$V4CKPT" --n-prompts 6 \
  > logs/phase09_rerun/V5_blocks.log 2>&1
grep -vE 'HTTP Request|urllib3|httpcore|INFO (tinker|__main__|scripts)' logs/phase09_rerun/V5_blocks.log | sed -n '1,120p'
```

**GATES (V5 PASS — ACCEPT only if ALL):**
- ECHO-ADVERSARY ≤ 0.30. **Hard.**
- The V4 policy's emitted blocks are REAL fix attempts of the SAME function (read them — not trivial
  echoes, not unrelated functions, not gutted bodies). **Hard, manual.**
- The V4 policy does NOT score systematically higher than warm-start by emitting reproductions (its
  identity-OK-but-no-improvement completions should sit ~0.5, not ~1.0).

**Branches:**
- ECHO > 0.30 or blocks are trivial/gutted → reward still has a hack surface → iterate the identity
  gate / improvement-delta (e.g. raise `_MIN_TOKEN_RETENTION`, require Δ on the FLAGGED dimension
  specifically). Do NOT authorize the 500-step run.
- Pass → V5 PASS, continue to V6.

---

### V6 — Codegen no-regression pre-check (RLEV-01 alignment)

**Purpose:** ensure FIX 2's template credit is ADDITIVE — it must not relax the gate so broken standalone
PHP now passes (which would let RL degrade base codegen undetected).

```bash
# 6a. Unit-level: standalone PHP scoring unchanged + broken still rejected.
$PY -m pytest tests/test_reward_fix2_gen.py -q     # all pass
$PY - <<'EOF'
from scripts.rl_rollouts import _is_valid_wp_php
assert _is_valid_wp_php("<?php function f(){ return 1; }") is True      # standalone unchanged
assert _is_valid_wp_php("<?php function f(){ return ") is False         # broken standalone still rejected
assert _is_valid_wp_php("<?php $x = ;") is False                        # broken still rejected
print("V6a additive-guard checks PASS")
EOF
```
```bash
# 6b. wp-bench codegen spot subset on the V4 checkpoint vs the v1.2 SFT baseline.
#     Locate the wp-bench runner (eval/ + scripts/_wpbench_* helpers; see STATE.md "wp-bench" /
#     04.4 REVL-04 chain a1cc63a). Run a small subset (e.g. WPBENCH_LIMIT=30) on BOTH the V4 policy
#     and the v1.2 SFT baseline served via vLLM; the V4 codegen score must NOT regress materially vs
#     baseline. (This is the same gate Phase 10 RLEV-01 enforces in full.)
```

**GATES (V6 PASS):**
- 6a: standalone-PHP validity identical to pre-FIX-2; broken PHP still rejected. **Hard.**
- 6b: V4 codegen wp-bench spot score ≥ v1.2 SFT baseline within noise (no material regression). **Hard.**

**Branches:**
- 6a fails → FIX 2 over-relaxed the gate → tighten `_neutralize_template_directives` so it cannot mask a
  broken PHP scaffold. Re-run V1 gen tests.
- 6b regresses → codegen damage from RL; the wp_gen replay protection (now live via FIX 2) may need
  re-weighting, or RL LR/steps revisited. Flag for the Phase 10 / reward-tuning discussion. Do NOT
  authorize the full run.

---

## 6. Overall acceptance + what comes after

**ALL of V2–V6 PASS →** the reward fixes are verified hack-resistant + non-regressing. Surface the V2–V6
results to Dr. Li and request authorization for a full RL re-spend:
- 2 seeds (42 / 7), `_launch_post81_rerun.sh 42 seedA` + `... 7 seedB` (those use --total-steps 500,
  --checkpoint-every 50), OOM guard armed, monitored as in J.1–J.4.
- Gate at step 50 on the JUDGE-axis controlled eval (V2 harness), NOT the live slope (the live slope is
  too noisy — see J.6; the negative-control method in V2 is the instrument).

**ANY V2–V6 FAIL →** follow that step's branch; do not proceed to the 500-step spend.

---

## 7. Known caveats (carry into V5 tuning)

1. **Rubric sensitivity limits the correctness signal.** The improvement-delta depends on
   `_extract_verifiable_signals` actually crediting the fix. Observed: `esc_html` alone did NOT move
   D2_security (stayed 3.0); only a fuller nonce+`wp_unslash`+`sanitize_text_field`+`esc_html` fix moved
   the rubric (70.2→100). So some genuine fixes may be UNDER-credited. If V5 shows real fixes scoring low,
   the limiter is the rubric, not the gate — consider scoring Δ on the SPECIFICALLY FLAGGED dimension
   rather than overall.
2. **Verification discipline (this bit us 3×).** Trust DATA over function-level scores: read raw
   completions (`_dump_corrected_blocks.py`), don't infer from a synthetic probe. The gen `parseable=0`,
   the step-30 "weak positive," and the "model is hacking" claims were all wrong until the actual data was
   read. V5's manual block read is mandatory for this reason.
3. **gen `critique_text` is empty** in the loaded judge pool (consistency is scored without it). Separate
   pre-existing issue; out of scope for V2–V6 but note it if consistency behaves oddly.
4. **Each RL launch is a fresh warm-start (NO resume).** A re-run starts from the v1.2 savestate at step 0.
5. **`.env` carries ANTHROPIC_API_KEY** — always `unset` it (per §2) so the $0 local-judge path can never
   leak to paid API.

## 8. Quick reference — one-liners

```bash
# regression suite
.venv-tinker/bin/python -m pytest tests/test_reward_fix1_judge.py tests/test_reward_fix2_gen.py tests/test_rl_train.py tests/test_rl_train_integration.py tests/test_reward_pipeline.py tests/test_rl_rollouts_reward_shape.py -q -k "not test_lora_config"
# offline hack-probe (no Tinker)
.venv-tinker/bin/python - <<'EOF'
from scripts.rl_rollouts import _fix_score_from_completion, _is_valid_wp_php
o="<?php\nfunction render($x){ $q=$_GET['q']; echo $q; }"
f=lambda p:"x\n```php\n"+p+"\n```"
for n,p in [("echo","<?php echo 'hi';"),("gut","<?php function render($x){ return 1; }"),("repro",o),("fix","<?php function render($x){ $q=sanitize_text_field(wp_unslash($_GET['q'])); echo esc_html($q); }")]:
    print(n, round(_fix_score_from_completion(f(p),o),3))
EOF
# stop an RL run
kill $(cat output/rl_checkpoints/metrics/rl_run.V4.pid)
```
