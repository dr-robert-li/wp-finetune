# Phase 09 — V2–V6 Verification Results

Branch `phase10-execution`. Executed 2026-06-26. Baselines pre-verified: 118 regression tests green,
offline hack-probe exact (echo 0.25 / gut 0.25 / repro 0.5 / fix 0.587), judges UP.

---

## V2 — Controlled judge-axis eval under NEW reward + echo-adversary — **PASS**

Harness `_check_judge_fixcorr.py` edited: pass `_judge_original_code(prompts[i//gsz])` into
`_fix_score_from_completion` (NEW correctness-pressured path), added `len(comps)==len(prompts)*gsz`
assert, 0.5-tier histogram, one-real-pair PAIR-CHECK, deterministic echo-adversary.

Run: `--fixed-50 tinker://3207bc66...step-50 --n-prompts 20 --group-size 2 --temperature 0.2`

| policy | n | mean_fixcorr | tiers 0.0/0.25/0.5/hi | Δ vs warm |
|---|---|---|---|---|
| warm-start | 40 | 0.2875 | 0/34/6/0 | — |
| fixed-step50 | 40 | 0.3750 | 0/20/20/0 | +0.0875 |
| stale-step50 (ctrl) | 40 | 0.3937 | 0/17/23/0 | +0.1063 |
| **ECHO-ADVERSARY** | — | **0.2500** | — | gate ≤0.30 |

Gates:
- **Gate 1 (echo ≤0.30): PASS** — 0.25. Isolation hack closed LIVE on sampled completions. (Only STOP branch; clean.)
- **Gate 2 (no saturation): PASS** — zero hi-tier, genuine 0.25/0.5 spread. Warm-vs-step50 gap compressed
  from old-reward J.7 (fixed +0.169→+0.0875, stale +0.219→+0.106).
- **Gate 3 (stale not >> warm / compression): PASS (spirit)** — confound halved (+0.219→+0.106).

Notes (advisor-reviewed):
- Printed harness VERDICT "CONFOUNDED" is DISREGARDED — it answers the harness's *original* question
  ("did learning happen", 0.03 threshold), not the V2 gate.
- Honest phrasing: **stale ≈ fixed** (0.394 vs 0.375 on n=40 ≈ 3 completions = noise), both modestly >
  warm. NOT "negative control beats real run." The warm-vs-step50 gap (≈17 completions) is real but is a
  reproduction-rate difference, capped at 0.5 — benign.
- **Pairing confirmed healthy**: PAIR-CHECK → prompts[0] `get_rest_routes()` orig extracted (543 chars);
  0.5 cluster grows monotonically with training (warm 6 → fixed 20 → stale 23) — impossible under broken
  pairing. This is the only thing V2 structurally needed.
- **Carry-forward**: >0.5 (genuine-fix) tier is COMPLETELY UNEXERCISED here — expected (old-isolation-reward
  policies echo/reproduce, no fix incentive). So V2 proves measurement + hack-resistance but NOT genuine-fix
  crediting. **V4 is the first test of the hi-tier.** If V4 `fix_correctness_mean` pins ~0.5 and never
  spreads above, that is handover caveat #1 (rubric under-credits real fixes), NOT FIX-1-not-plumbed.

---

## V3 — Gen-axis liveness under FIX 2 — **PASS** (after a NEW live-bug fix)

The handover's brittle 3-policy harness (`_check_step50_vs_warmstart.py`) hung HARD three times
(`STAT=Sl`, 5s CPU / 1800s, wchan `poll_schedule_timeout`) — `_generate_completions` loops per-prompt
with a blocking `sample().result()` and **no timeout**, so a single pathological prompt's hung Tinker
request froze the whole run. Replaced with a robust per-prompt-timeout probe (`scripts/_v3_liveness.py`)
that skips hangs and scores warm-start gen completions old-gate vs new-gate.

**Root-cause bug found (read-the-data discipline, handover caveat #2):** first liveness run = **0/24
parseable by BOTH gates**. Dumping raw completions (`scripts/_v3_dump.py`) showed every gen completion
ends with the literal chat EOS marker `<|im_end|>`; `php -l` fails identically — *"unexpected token '<',
expecting end of file."* `_generate_completions` decoded with `tok.decode(tokens)` (no
`skip_special_tokens`), so the marker leaked into the TEXT. Judge path (V2) was spared only because
fenced ```php extraction drops anything after the closing fence; **bare-code gen completions carried the
marker into the scored PHP → every gen reward zeroed.** A confirmed (likely dominant) cause of the dead
gen gradient — independent of FIX 2's Elementor-template target.

**Fix (committed):** `scripts/rl_rollouts.py:1220` → `tok.decode(tokens, skip_special_tokens=True)`.
TEXT-only strip; `.tokens`/`.logprobs` keep the EOS token (GSPO IS ratio load-bearing). Grep confirmed no
`.completion` consumer keys on the marker (single `tok.decode` call site). Faithful test double + 2 new
regression tests (`test_reward_fix2_gen.py`): leaked-marker-is-fatal-when-unstripped, and
`_generate_completions` strips text but retains tokens. Suite **120 green** (was 118).

**Re-run after fix** (`logs/.../V3_liveness2.log`, decode-strip mirrored in the probe):

| gate | parseable | mean_reward |
|---|---|---|
| OLD `_is_parseable_php` | 16/24 | 0.0416 |
| NEW `_is_valid_wp_php` | 16/24 | 0.0416 |

Gates: warm parseable(new) > 0 → **PASS (16, was 0)** · ≥1 nonzero gen mean_reward → **PASS (0.0416, was
0)** · FIX2 additive (new ≥ old) → **PASS (16≥16)**.

Notes:
- FIX-2 template-specific credit = 0 in this draw (no Elementor templates among the 12 sampled prompts);
  that path is covered by offline unit tests (`test_elementor_template_valid`, `test_bare_method_body_template_valid`).
  The live win here is the **decode fix** (0→16 parseable), not template-awareness.
- 8/24 still fail parse = genuinely broken/truncated (384-tok cut) PHP — correctly zeroed.
- **V2 NOT re-run** (decision, not gap): the judge path extracts the fenced block; `<|im_end|>` fell after
  the closing fence and was already excluded, so fix_scores are identical. The decode fix can only RAISE
  parseable rates, never lower them — it cannot invalidate a V2 PASS.
- **V4 is where this fix is validated live**: `frac_groups_all_zero` must now drop below 0.375 (gen
  contributes). Watch it specifically.
