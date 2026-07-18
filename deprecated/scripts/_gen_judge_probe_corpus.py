#!/usr/bin/env python3
"""Generate a fresh judge-completion corpus for the offline reward-shape probe.

$0 / LOCAL-vLLM ONLY. This helper NEVER imports or calls any hosted LLM vendor
API or one-shot CLI. It talks to an OpenAI-compatible **local vLLM** endpoint only.

What it produces
----------------
A JSONL corpus of judge completions, generated with the EXACT live-RL judge
prompt format and sampling, then counted "parseable" with the EXACT downstream
gate. The output is consumed by `scripts/_probe_rl_reward.py` (which reads a
`raw_text` field per line). The adaptive loop keeps generating until it has
`--target` parseable completions (default 300) or hits `--max-raw` (default 1200).

Faithfulness to the live RL run (so "300 parseable" == what the probe counts):
  - Prompts: loaded from the SAME source the live RL rollout uses,
    `scripts.tinker_rl_data.load_rl_prompts("judge")`
    (-> data/rl_prompts/wp_judge_train.jsonl), then passed through
    `rl_rollouts._augment_judge_prompt` (the critique-then-fix output contract
    appended at rollout time, rl_rollouts.py line ~666).
  - Sampling: group_size=4 (G), temperature=1.0, max_tokens=1536
    (= rl_rollouts.JUDGE_MAX_NEW_TOKENS, the judge rollout cap). The live path
    calls sample(num_samples=G) at temp 1.0 with NO per-sample seed, so G
    independent stochastic draws faithfully reproduce it.
  - Parseable gate: composed BYTE-FOR-BYTE with the rl_rollouts fix_correctness
    parse-gate (rl_rollouts.py lines 718-719):
        corrected = _extract_corrected_php(completion_text)
        parseable = _is_parseable_php(corrected)
    Both helpers are imported directly from rl_rollouts (no reinvention).
  - Server call: reuses eval/eval_judge.py `_judge_create`, which carries the
    MANDATORY RC-A guard `extra_body={"chat_template_kwargs":
    {"enable_thinking": False}}` (imported LAZILY inside the server path so the
    offline --dry-run / --help do not need scipy/openai/dgx_toolbox).

Usage (after a vLLM judge server is up; e.g. served model "wp_judge" on :8000):
    # OFFLINE sanity check first (no GPU, no server):
    REWARD_SKIP_PHPCS_ASSERT=1 python3 scripts/_gen_judge_probe_corpus.py --dry-run

    # Real generation against the served model:
    REWARD_SKIP_PHPCS_ASSERT=1 python3 scripts/_gen_judge_probe_corpus.py \
        --base-url http://localhost:8000/v1 --target 300

Then run the probe on the corpus it printed:
    REWARD_SKIP_PHPCS_ASSERT=1 python3 scripts/_probe_rl_reward.py \
        --completions data/rl_probe/judge_probe_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make the repo root importable regardless of CWD (mirrors _probe_rl_reward.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The live security gate's phpcs assert is about training; this offline helper
# never touches it. Set before any reward import, matching _probe_rl_reward.py.
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")

# rl_rollouts top-level imports are stdlib + numpy only (verified), so importing
# the two parse helpers here is offline-safe. _extract_corrected_php lazily
# imports eval.output_parsers.extract_php_code INSIDE the function, so this
# import does not pull in scipy/openai/dgx_toolbox.
from scripts.rl_rollouts import (  # noqa: E402
    JUDGE_MAX_NEW_TOKENS,
    _augment_judge_prompt,
    _extract_corrected_php,
    _is_parseable_php,
)
from scripts.tinker_rl_data import load_rl_prompts  # noqa: E402

DEFAULT_OUT = os.path.join("data", "rl_probe", "judge_probe_corpus.jsonl")
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "wp_judge"


# ---------------------------------------------------------------------------
# Parseable gate — composed EXACTLY as rl_rollouts.collect_rollouts does
# (rl_rollouts.py lines 716-724). DO NOT loosen/tighten: the probe counts the
# same composition, so this is what makes "300 parseable" here == the probe's.
# ---------------------------------------------------------------------------
def is_parseable_completion(completion_text: str) -> bool:
    """True iff a judge completion yields a usable, syntactically-valid PHP fix.

    Mirrors rl_rollouts fix_correctness parse-gate byte-for-byte:
        corrected = _extract_corrected_php(completion)
        parseable = _is_parseable_php(corrected)
    """
    corrected = _extract_corrected_php(completion_text)
    return _is_parseable_php(corrected)


# ---------------------------------------------------------------------------
# Prompt pool: live RL rollout source, augmented with the judge output contract
# ---------------------------------------------------------------------------
def build_prompt_pool(override_path: str | None = None) -> list[dict]:
    """Load the wp_judge prompt pool and apply _augment_judge_prompt to each.

    By default this is the EXACT source the live RL run uses
    (load_rl_prompts("judge")). --prompts PATH overrides it with a JSONL of
    {"messages": [...]} (or {"prompt": "..."}) rows.

    Each returned item carries a stable "prompt_id" (pool index) so cycling the
    pool keeps prompt provenance even as group_ids advance.
    """
    if override_path:
        pool = _load_jsonl_prompts(override_path)
    else:
        # Dependency-free manual JSONL load inside tinker_rl_data (no tinker/GPU).
        pool = load_rl_prompts("judge")

    augmented = []
    for idx, item in enumerate(pool):
        # _augment_judge_prompt mutates+returns; copy messages so we don't alter
        # the loader's objects across reuse. It is idempotent regardless.
        msgs = [dict(m) for m in (item.get("messages") or [])]
        clone = {**item, "messages": msgs}
        _augment_judge_prompt(clone)
        clone["prompt_id"] = idx
        augmented.append(clone)
    return augmented


def _load_jsonl_prompts(path: str) -> list[dict]:
    """Load an override prompt JSONL: user-turn-only, same shape as the loader."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"--prompts file not found: {path}")
    prompts: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if msgs:
                user_msgs = [m for m in msgs if m.get("role") == "user"]
                if user_msgs:
                    prompts.append({"messages": user_msgs})
                continue
            if row.get("prompt"):
                prompts.append(
                    {"messages": [{"role": "user", "content": str(row["prompt"])}]}
                )
    if not prompts:
        raise ValueError(f"--prompts file had no usable prompts: {path}")
    return prompts


def _prompt_messages(item: dict) -> list[dict]:
    """Return the user-turn message list for the OpenAI chat call."""
    msgs = item.get("messages") or []
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    if user_msgs:
        return user_msgs
    return [{"role": "user", "content": str(item.get("prompt", ""))}]


# ---------------------------------------------------------------------------
# Server (LAZY imports: keep openai/scipy/dgx_toolbox out of the offline path)
# ---------------------------------------------------------------------------
def _build_client(base_url: str, api_key: str):
    import openai  # noqa: PLC0415 — lazy: offline --dry-run/--help must not need it

    return openai.OpenAI(base_url=base_url, api_key=api_key or "EMPTY")


def _detect_model(client, fallback: str = DEFAULT_MODEL) -> str:
    """Autodetect the served model from /v1/models; fall back to `wp_judge`.

    We do NOT reuse eval_judge._detect_model because it falls back to
    'openai/qwen3-wp'; requirement 1 wants the wp_judge default here.
    """
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:  # noqa: BLE001
        pass
    return fallback


def _generate_one(client, model, messages, max_tokens, temperature, retries, backoff):
    """One completion via eval_judge._judge_create (preserves the RC-A guard).

    Retries transient errors with bounded backoff. Returns the completion text,
    or raises after exhausting retries (caller logs+skips so one bad call never
    crashes the whole run).
    """
    from eval.eval_judge import _judge_create  # noqa: PLC0415 — lazy server path

    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = _judge_create(
                client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 — bounded retry on transient errors
            last_exc = e
            if attempt < retries:
                sleep_s = backoff * (2 ** attempt)
                print(
                    f"  [retry {attempt + 1}/{retries}] {type(e).__name__}: {e} "
                    f"-> sleeping {sleep_s:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_s)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def _count_existing(path: str) -> tuple[int, int, int]:
    """Return (raw_count, parseable_count, max_group_id) from an existing file."""
    if not os.path.exists(path):
        return 0, 0, -1
    raw = 0
    parseable = 0
    max_gid = -1
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001 — tolerate a partial trailing line
                continue
            raw += 1
            if d.get("parseable"):
                parseable += 1
            gid = d.get("group_id")
            if isinstance(gid, int) and gid > max_gid:
                max_gid = gid
    return raw, parseable, max_gid


# ---------------------------------------------------------------------------
# Dry-run self-test (NO server, NO GPU, NO network)
# ---------------------------------------------------------------------------
_DRY_VALID_PHP_COMPLETION = (
    "The code is missing input sanitization and a nonce check.\n\n"
    "Then FIX the code: here is the corrected version.\n\n"
    "```php\n"
    "<?php\n"
    "function wp_demo_save( $post_id ) {\n"
    "    if ( ! isset( $_POST['demo_nonce'] ) ) {\n"
    "        return;\n"
    "    }\n"
    "    $value = sanitize_text_field( wp_unslash( $_POST['demo'] ) );\n"
    "    update_post_meta( $post_id, '_demo', $value );\n"
    "}\n"
    "```\n"
)

_DRY_CORRECTED_TAG_COMPLETION = (
    "Critique: the function lacks capability checks.\n"
    "<corrected_code>\n"
    "<?php\n"
    "if ( current_user_can( 'edit_posts' ) ) {\n"
    "    do_action( 'demo_hook' );\n"
    "}\n"
    "</corrected_code>\n"
)

_DRY_PROSE_ONLY_COMPLETION = (
    "Overall this code looks reasonable. I would rate it 8/10 for security and "
    "9/10 for WPCS compliance. Could you clarify which file this belongs to "
    "before I suggest a corrected version?"
)


def _run_dry_run(args) -> int:
    print("=" * 72)
    print("DRY-RUN (offline): NO server, NO GPU, NO network will be contacted.")
    print("=" * 72)

    # 1) Prompt pool + augmented format
    pool = None
    synthesized = False
    try:
        pool = build_prompt_pool(args.prompts)
        print(f"\nPrompt pool loaded: {len(pool)} wp_judge prompts "
              f"(source: {args.prompts or 'load_rl_prompts(\"judge\")'}).")
    except Exception as e:  # noqa: BLE001
        synthesized = True
        print(f"\n[dry-run] prompt pool could not load offline ({type(e).__name__}: {e}).")
        print("[dry-run] SYNTHESIZING 2 example prompts so --dry-run still works.")
        pool = [
            _augment_judge_prompt(
                {"messages": [{"role": "user", "content":
                    "<wp_judge> Evaluate this WordPress code:\n```php\n<?php echo $_GET['x'];\n```"}],
                 "prompt_id": 0}
            ),
            _augment_judge_prompt(
                {"messages": [{"role": "user", "content":
                    "<wp_judge> Evaluate this WordPress code:\n```php\n<?php add_action('init', 'foo');\n```"}],
                 "prompt_id": 1}
            ),
        ]

    print("\n--- ONE fully-augmented judge prompt (what gets sent to the server) ---")
    example_msgs = _prompt_messages(pool[0])
    for m in example_msgs:
        print(f"[role={m.get('role')}]")
        print(m.get("content", ""))
    # Confirm the output contract was appended.
    joined = "\n".join(m.get("content", "") for m in example_msgs)
    contract_present = "```php fenced block" in joined
    print(f"\n[check] output-contract appended by _augment_judge_prompt: {contract_present}")
    if synthesized:
        print("[note] prompts above are SYNTHETIC (offline fallback), not the real pool.")

    # 2) Parseable-counter self-test on synthetic strings (the COMPOSED gate)
    print("\n--- parseable-counter self-test (composed gate: "
          "_is_parseable_php(_extract_corrected_php(text))) ---")
    cases = [
        ("valid-PHP fix (```php fence)", _DRY_VALID_PHP_COMPLETION, True),
        ("valid-PHP fix (<corrected_code>)", _DRY_CORRECTED_TAG_COMPLETION, True),
        ("prose-only non-answer", _DRY_PROSE_ONLY_COMPLETION, False),
    ]
    all_ok = True
    for label, text, expected in cases:
        got = is_parseable_completion(text)
        status = "OK" if got == expected else "FAIL"
        if got != expected:
            all_ok = False
        print(f"  [{status}] {label}: expected={expected} got={got}")

    if not all_ok:
        print("\nDRY-RUN FAILED: parseable-counter did not return expected bools.", file=sys.stderr)
        return 1
    print("\nDRY-RUN PASSED: prompt format + parseable-counter behave as expected.")
    print("(If php is unavailable, _is_parseable_php fails OPEN -> True; here php is "
          "expected to be installed so the prose-only case returns False.)")
    return 0


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate a judge-completion corpus for the offline reward-shape probe "
            "($0 LOCAL vLLM only; never calls any hosted LLM vendor API or CLI)."
        )
    )
    p.add_argument("--base-url", default=None,
                   help=f"vLLM OpenAI base URL (else $EVAL_JUDGE_BASE_URL else {DEFAULT_BASE_URL}).")
    p.add_argument("--api-key", default=None,
                   help="Dummy API key for the local endpoint (default 'EMPTY'; vLLM ignores it).")
    p.add_argument("--model", default=None,
                   help=f"Served model name (else autodetect /v1/models else '{DEFAULT_MODEL}').")
    p.add_argument("--prompts", default=None,
                   help="Override the wp_judge prompt source (JSONL of {messages|prompt}).")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"Output JSONL path (append-mode). Default: {DEFAULT_OUT}")
    p.add_argument("--group-size", type=int, default=4,
                   help="Completions per prompt (GRPO group size G). Default 4 (matches live).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature. Default 1.0 (matches live rollout).")
    p.add_argument("--max-tokens", type=int, default=JUDGE_MAX_NEW_TOKENS,
                   help=f"max_tokens per completion. Default {JUDGE_MAX_NEW_TOKENS} "
                        f"(= rl_rollouts.JUDGE_MAX_NEW_TOKENS, the judge rollout cap).")
    p.add_argument("--target", type=int, default=300,
                   help="Stop once this many PARSEABLE completions exist. Default 300.")
    p.add_argument("--max-raw", type=int, default=1200,
                   help="Hard stop on total raw completions (safety cap). Default 1200.")
    p.add_argument("--resume", action="store_true",
                   help="Count existing parseable rows in --out and continue appending.")
    p.add_argument("--progress-every", type=int, default=20,
                   help="Print a live progress line every N completions. Default 20.")
    p.add_argument("--retries", type=int, default=4,
                   help="Bounded retries per completion on transient errors. Default 4.")
    p.add_argument("--backoff", type=float, default=1.0,
                   help="Base backoff seconds (exponential). Default 1.0.")
    p.add_argument("--max-consecutive-failures", type=int, default=None,
                   help="Abort if this many completions fail in a row (default 2*group_size). "
                        "Converts a wrong model name / down endpoint into a fast, "
                        "diagnosable abort instead of an infinite skip loop.")
    p.add_argument("--dry-run", action="store_true",
                   help="OFFLINE self-test: print one augmented prompt + run the "
                        "parseable-counter on synthetic strings; contact NO server.")
    args = p.parse_args(argv)

    if args.dry_run:
        return _run_dry_run(args)

    # ---- live generation (server required from here on) ----
    base_url = args.base_url or os.environ.get("EVAL_JUDGE_BASE_URL") or DEFAULT_BASE_URL
    api_key = args.api_key or "EMPTY"

    pool = build_prompt_pool(args.prompts)
    if not pool:
        print("ERROR: empty wp_judge prompt pool.", file=sys.stderr)
        return 1
    print(f"Loaded wp_judge prompt pool: {len(pool)} prompts "
          f"(source: {args.prompts or 'load_rl_prompts(\"judge\")'}).")

    client = _build_client(base_url, api_key)
    model = args.model or _detect_model(client)
    print(f"Endpoint: {base_url}  |  served model: {model}")
    print(f"Sampling: G={args.group_size}, temperature={args.temperature}, "
          f"max_tokens={args.max_tokens}")
    print(f"Targets: parseable>={args.target} OR raw>={args.max_raw}")

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    raw_count = 0
    parseable_count = 0
    next_group_id = 0
    if args.resume:
        raw_count, parseable_count, max_gid = _count_existing(out_path)
        next_group_id = max_gid + 1
        print(f"[resume] existing: raw={raw_count} parseable={parseable_count} "
              f"-> next group_id={next_group_id}")

    if parseable_count >= args.target:
        print(f"[resume] target already met (parseable={parseable_count} >= {args.target}). Nothing to do.")
        _print_summary(out_path, raw_count, parseable_count)
        return 0

    pool_idx = 0
    n_prompts = len(pool)
    max_consec = (
        args.max_consecutive_failures
        if args.max_consecutive_failures is not None
        else 2 * args.group_size
    )
    consecutive_failures = 0

    with open(out_path, "a", encoding="utf-8") as out_f:
        while parseable_count < args.target and raw_count < args.max_raw:
            item = pool[pool_idx % n_prompts]
            pool_idx += 1
            prompt_id = item.get("prompt_id", pool_idx - 1)
            messages = _prompt_messages(item)
            group_id = next_group_id
            next_group_id += 1

            # G independent stochastic draws at temp 1.0 == the live
            # sample(num_samples=G) path (which also does not seed per sample).
            for sample_idx in range(args.group_size):
                if raw_count >= args.max_raw:
                    break
                try:
                    text = _generate_one(
                        client, model, messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        retries=args.retries,
                        backoff=args.backoff,
                    )
                except Exception as e:  # noqa: BLE001 — never crash the whole run
                    print(f"  [skip] group={group_id} sample={sample_idx} "
                          f"failed after retries: {type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
                    consecutive_failures += 1
                    if consecutive_failures >= max_consec:
                        print(f"\nABORT: {consecutive_failures} consecutive failures "
                              f"(endpoint={base_url}, model={model}). This usually means a "
                              f"wrong --model name, a down/unreachable endpoint, or no served "
                              f"model. Fix and re-run (use --resume to keep what was written).",
                              file=sys.stderr, flush=True)
                        _print_summary(out_path, raw_count, parseable_count)
                        return 1
                    continue

                try:
                    parseable = is_parseable_completion(text)
                except Exception as e:  # noqa: BLE001 — a gate hiccup must not crash
                    print(f"  [skip] group={group_id} sample={sample_idx} "
                          f"parse-gate error: {type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
                    consecutive_failures += 1
                    if consecutive_failures >= max_consec:
                        print(f"\nABORT: {consecutive_failures} consecutive failures "
                              f"(parse-gate). This is unexpected offline-style behavior; "
                              f"check the php binary / eval.output_parsers import.",
                              file=sys.stderr, flush=True)
                        _print_summary(out_path, raw_count, parseable_count)
                        return 1
                    continue

                consecutive_failures = 0

                row = {
                    "raw_text": text,
                    "group_id": group_id,
                    "sample_idx": sample_idx,
                    "prompt_id": prompt_id,
                    "parseable": parseable,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()

                raw_count += 1
                if parseable:
                    parseable_count += 1

                if raw_count % max(1, args.progress_every) == 0:
                    fail_rate = (raw_count - parseable_count) / raw_count if raw_count else 0.0
                    print(f"  progress: raw={raw_count} parseable={parseable_count} "
                          f"parse_fail_rate={fail_rate:.3f} "
                          f"(target {args.target}, cap {args.max_raw})", flush=True)

    if parseable_count < args.target:
        print(f"\nWARNING: hit max-raw cap ({raw_count}) before reaching target "
              f"({parseable_count}/{args.target} parseable). The model's parse-fail "
              f"rate may be high (this is itself a signal). Re-run with --resume "
              f"and a higher --max-raw if you need the full target.", file=sys.stderr)

    _print_summary(out_path, raw_count, parseable_count)
    return 0


def _print_summary(out_path: str, raw_count: int, parseable_count: int) -> None:
    fail_rate = (raw_count - parseable_count) / raw_count if raw_count else 0.0
    abspath = os.path.abspath(out_path)
    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"corpus path     : {abspath}")
    print(f"raw completions : {raw_count}")
    print(f"parseable       : {parseable_count}")
    print(f"parse-fail rate : {fail_rate:.3f}")
    print("\nReady-to-run probe command:")
    print(f"  REWARD_SKIP_PHPCS_ASSERT=1 python3 scripts/_probe_rl_reward.py "
          f"--completions {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
