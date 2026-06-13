"""Phase 04.4 Plan 02 Task 1 — single vLLM serve session for the v4 winner staging model.

Serves models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4/ once and captures every
downstream-gate input in that single GPU session (D-V4-03 capture mechanism):

  1. judge_val_responses.jsonl   — REVL-01A Spearman + REVL-07 confusion source
  2. sentinel_responses.jsonl    — REVL-05 sentinel (24 held-out invalid-PHP prompts)
  3. reasoning_samples.jsonl     — REVL-03 / REVL-08 reasoning chain samples (cot+ctf)
  4. gen_samples.jsonl           — REVL-06 fix-correctness samples (ctf stream)
  5. eval_gen_results.json       — REVL-02 phpcs_pass_rate measured on v4-winner
  6. capture_manifest.json       — provenance: served id, model dir, stream counts, timestamps

All judge captures use enable_thinking=False (RC-A fix) via extra_body so <think> blocks
never corrupt the judge JSON.  eval_gen handles think-stripping post-hoc.

Schema for judge_val + sentinel: {index, response} with a __provenance__ header row —
matches what eval_judge.py --responses-jsonl and check_invalid_php_sentinel.py consume.
Index = position in the wp_judge-filtered list (mirrors eval_judge._run_eval_reasoning).

Schema for reasoning_samples + gen_samples: {example_idx, task_type, prompt, response,
model_scores} with a __provenance__ header row — matches capture_reasoning_responses.py
output (what REVL-03/REVL-06/REVL-08 aggregators consume).

Crash-safe reuse: if a non-empty capture already exists for a stream, skip re-capturing it.

Usage:
  python scripts/_04.4_v4winner_serve_capture.py
  python scripts/_04.4_v4winner_serve_capture.py --staging-dir models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Defaults (parameterized — no nolmhead hardcodes)
# ---------------------------------------------------------------------------
DEFAULT_STAGING_DIR = "models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4"
SERVED_MODEL_NAME = "wp-30_70"
PORT = 8021
GPU_MEM_UTIL = 0.55
CONTAINER_NAME = "wp-v4winner-capture-vllm"
OUT_DIR = ROOT / "output" / "eval_reasoning_v4_winner"
DATASET = "data/reasoning_dataset/openai_val.jsonl"
SENTINEL_DATASET = "data/reasoning_dataset/invalid_php_sentinel.jsonl"
LOG_DIR = ROOT / "logs" / "phase4.4"
LOG_PATH = LOG_DIR / "v4winner_serve_capture.log"

# Minimum free GPU memory (GiB) before serving (Pitfall-6 memory floor guard)
MIN_FREE_GPU_GIB = 70.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str, log_fh) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


# ---------------------------------------------------------------------------
# Preflight helpers
# ---------------------------------------------------------------------------

def _preflight_staging(staging_dir: Path, log_fh) -> None:
    """Assert staging dir exists and merge_report passes basic anchor checks."""
    _log(f"Preflight: staging_dir={staging_dir}", log_fh)
    if not staging_dir.exists():
        _log(f"ABORT: staging dir not found: {staging_dir}", log_fh)
        sys.exit(2)
    # Check for merge_report — use merge_v4_winner (not nolmhead)
    merge_report_path = ROOT / "output" / "merge_v4_winner" / "merge_report.json"
    if merge_report_path.exists():
        rpt = json.loads(merge_report_path.read_text())
        anchors_ok = rpt.get("anchors_all_pass")
        _log(f"Preflight: merge_report anchors_all_pass={anchors_ok} "
             f"(report={merge_report_path})", log_fh)
        if anchors_ok is False:
            _log("ABORT: anchors_all_pass=False in merge_report.", log_fh)
            sys.exit(2)
    else:
        _log(f"WARNING: merge_report not found at {merge_report_path} — skipping anchor check",
             log_fh)
    _log("Preflight: staging dir OK", log_fh)


def _preflight_memory(log_fh) -> None:
    """Check for heavy GPU apps and warn if free memory is below floor."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                val = line.strip()
                if val and val != "[N/A]" and val.isdigit():
                    free_gib = int(val) / 1024.0
                    _log(f"GPU free memory: {free_gib:.1f} GiB (floor={MIN_FREE_GPU_GIB} GiB)",
                         log_fh)
                    if free_gib < MIN_FREE_GPU_GIB:
                        _log(f"WARNING: free GPU memory {free_gib:.1f} GiB < "
                             f"{MIN_FREE_GPU_GIB} GiB floor — OOM risk", log_fh)
                    return
    except Exception as e:  # noqa: BLE001
        pass
    _log("NOTE: nvidia-smi memory check not available (normal for GB10 unified-memory)", log_fh)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _wp_judge_rows(rows: list[dict]) -> list[tuple[int, dict]]:
    """Return (filtered_index, row) for rows whose first user message starts with <wp_judge>.

    Index discipline: position in the filtered list (0..N-1), not in the full file.
    This matches eval_judge._run_eval_reasoning enumeration for correct Spearman pairing.
    """
    result = []
    idx = 0
    for row in rows:
        user_msg = next(
            (m["content"] for m in row.get("messages", []) if m["role"] == "user"), ""
        )
        if user_msg.startswith("<wp_judge>"):
            result.append((idx, row))
            idx += 1
    return result


def _stream_rows(rows: list[dict], streams: set) -> list[tuple[int, dict]]:
    """Return (file_index, row) for rows matching the given stream set."""
    result = []
    for i, row in enumerate(rows):
        if row.get("metadata", {}).get("stream") in streams:
            result.append((i, row))
    return result


# ---------------------------------------------------------------------------
# Judge capture (enable_thinking=False via extra_body — RC-A fix)
# ---------------------------------------------------------------------------

def _query_judge(client, served_model: str, messages: list[dict],
                 max_tokens: int, enable_thinking_fallback: bool,
                 _disable_thinking_enabled: list) -> str:
    """Query the judge endpoint with enable_thinking=False where supported."""
    kwargs: dict = {
        "model": served_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if _disable_thinking_enabled[0]:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    except Exception as e:
        emsg = str(e)
        if _disable_thinking_enabled[0] and (
            "enable_thinking" in emsg or "chat_template" in emsg or "template" in emsg
        ):
            # Model template doesn't support enable_thinking — fall back gracefully (RC-A)
            _disable_thinking_enabled[0] = False
            del kwargs["extra_body"]
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        raise


def _capture_judge_responses(
    client,
    served_model: str,
    examples: list[tuple[int, dict]],
    out_path: Path,
    provenance: str,
    log_fh,
    max_tokens: int = 2048,
    stream_name: str = "judge",
) -> int:
    """Capture judge responses in {index, response} schema.

    Writes provenance header + one row per example.
    Returns number of rows written (excl. header).
    """
    from eval.eval_judge import parse_judge_response  # noqa: F401 (presence check)

    _disable_thinking_enabled = [True]  # mutable flag for graceful fallback

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as fh:
        header = {
            "__provenance__": provenance,
            "dataset": stream_name,
            "served_model": served_model,
            "enable_thinking": False,
            "max_tokens": max_tokens,
            "n": len(examples),
        }
        fh.write(json.dumps(header) + "\n")
        for i, (idx, row) in enumerate(examples):
            msgs = [m for m in row.get("messages", []) if m["role"] == "user"]
            try:
                response = _query_judge(
                    client, served_model, msgs, max_tokens,
                    enable_thinking_fallback=True,
                    _disable_thinking_enabled=_disable_thinking_enabled,
                )
            except Exception as e:  # noqa: BLE001
                _log(f"[{stream_name}] gen error idx {idx}: {e}", log_fh)
                response = ""
            rec = {"index": idx, "response": response}
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if (i + 1) % 20 == 0:
                _log(f"[{stream_name}] captured {i+1}/{len(examples)}", log_fh)
    _log(f"[{stream_name}] captured {n} rows -> {out_path}", log_fh)
    return n


# ---------------------------------------------------------------------------
# Reasoning/gen samples capture (capture_reasoning_responses schema)
# ---------------------------------------------------------------------------

_THINK_STRIP = None


def _strip_think(text: str) -> str:
    global _THINK_STRIP
    if _THINK_STRIP is None:
        import re
        _THINK_STRIP = re.compile(r"<think>.*?</think>", re.DOTALL)
    return _THINK_STRIP.sub("", text).strip()


def _capture_reasoning_samples(
    client,
    served_model: str,
    examples: list[tuple[int, dict]],
    out_path: Path,
    provenance: str,
    log_fh,
    max_tokens: int = 2048,
    stream_name: str = "reasoning",
) -> int:
    """Capture reasoning/gen samples in capture_reasoning_responses schema.

    Schema per row: {example_idx, task_type, prompt, response, model_scores}
    Returns number of rows written (excl. header).
    """
    from eval.eval_judge import parse_judge_response

    out_path.parent.mkdir(parents=True, exist_ok=True)
    captured = []
    with open(out_path, "w") as fh:
        header = {
            "__provenance__": provenance,
            "stream": stream_name,
            "served_model": served_model,
            "max_tokens": max_tokens,
            "n": len(examples),
        }
        fh.write(json.dumps(header) + "\n")
        for i, (idx, row) in enumerate(examples):
            msgs = [m for m in row.get("messages", []) if m["role"] == "user"]
            task_type = row.get("metadata", {}).get("stream", "?")
            try:
                resp = client.chat.completions.create(
                    model=served_model,
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                response = resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                _log(f"[{stream_name}] gen error idx {idx}: {e}", log_fh)
                response = ""
            rec = {
                "example_idx": idx,
                "task_type": task_type,
                "prompt": msgs[0]["content"] if msgs else "",
                "response": response,
                "model_scores": parse_judge_response(response) if response else None,
            }
            captured.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 20 == 0:
                _log(f"[{stream_name}] captured {i+1}/{len(examples)}", log_fh)
    _log(f"[{stream_name}] captured {len(captured)} rows -> {out_path}", log_fh)
    return len(captured)


# ---------------------------------------------------------------------------
# Non-empty file check (crash-safe reuse)
# ---------------------------------------------------------------------------

def _is_nonempty_capture(path: Path) -> bool:
    """Return True if the path exists and has at least 2 lines (header + 1 data row)."""
    if not path.exists():
        return False
    lines = sum(1 for _ in open(path) if _.strip())
    return lines >= 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 04.4 Plan 02 Task 1 — v4 winner serve+capture")
    ap.add_argument("--staging-dir", default=DEFAULT_STAGING_DIR,
                    help="Path to the v4 staging model (relative to project root or absolute)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--gpu-mem-util", type=float, default=GPU_MEM_UTIL)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    staging_dir = Path(args.staging_dir) if Path(args.staging_dir).is_absolute() else ROOT / args.staging_dir
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    provenance = str(staging_dir)

    # Capture output paths
    judge_val_path = out_dir / "judge_val_responses.jsonl"
    sentinel_path = out_dir / "sentinel_responses.jsonl"
    reasoning_path = out_dir / "reasoning_samples.jsonl"
    gen_path = out_dir / "gen_samples.jsonl"
    eval_gen_path = out_dir / "eval_gen_results.json"
    manifest_path = out_dir / "capture_manifest.json"

    with open(LOG_PATH, "a") as log_fh:
        _log("=== _04.4_v4winner_serve_capture.py START ===", log_fh)
        _log(f"staging_dir: {staging_dir}", log_fh)
        _log(f"out_dir:     {out_dir}", log_fh)

        # --- Preflights ---
        _preflight_staging(staging_dir, log_fh)
        _preflight_memory(log_fh)

        # --- Load datasets ---
        ds_path = ROOT / DATASET
        sentinel_ds_path = ROOT / SENTINEL_DATASET
        _log(f"Loading dataset: {ds_path}", log_fh)
        val_rows = _load_dataset(ds_path)
        sentinel_rows = _load_dataset(sentinel_ds_path)
        _log(f"val rows: {len(val_rows)}  sentinel rows: {len(sentinel_rows)}", log_fh)

        # wp_judge filtered examples for judge_val and sentinel (index discipline)
        judge_val_examples = _wp_judge_rows(val_rows)
        _log(f"wp_judge filtered examples (judge_val): {len(judge_val_examples)}", log_fh)

        # Sentinel: all 24 rows are should_fail=True judge prompts — capture all in order
        # The sentinel checker uses same {index, response} schema with filtered index
        sentinel_examples = []
        sidx = 0
        for row in sentinel_rows:
            sentinel_examples.append((sidx, row))
            sidx += 1
        _log(f"sentinel examples: {len(sentinel_examples)}", log_fh)

        # Reasoning samples: cot+ctf streams
        reasoning_examples = _stream_rows(val_rows, {"cot", "ctf"})
        _log(f"reasoning_samples (cot+ctf): {len(reasoning_examples)}", log_fh)

        # Gen samples: ctf stream (CtF = critique-then-fix, fix-correctness)
        gen_examples = _stream_rows(val_rows, {"ctf"})
        _log(f"gen_samples (ctf): {len(gen_examples)}", log_fh)

        # --- Boot vLLM ---
        from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm, VllmBootTimeout

        endpoint = f"http://localhost:{args.port}/v1"
        import openai
        client = None
        served_identity = None

        try:
            _log(f"Booting vLLM: {CONTAINER_NAME} port={args.port} gpu_mem_util={args.gpu_mem_util}",
                 log_fh)
            boot_vllm(str(staging_dir), CONTAINER_NAME, args.port, args.gpu_mem_util)
            served_identity = wait_healthy(args.port, CONTAINER_NAME)
            _log(f"vLLM healthy; served_identity={served_identity}", log_fh)

            # --- Assert served identity ---
            if served_identity != SERVED_MODEL_NAME:
                _log(f"ABORT: served model is '{served_identity}', expected '{SERVED_MODEL_NAME}'",
                     log_fh)
                sys.exit(4)
            _log(f"Served identity OK: {served_identity}", log_fh)

            client = openai.OpenAI(base_url=endpoint, api_key="none")

            # ---- Capture 1: judge_val_responses ----
            if _is_nonempty_capture(judge_val_path):
                n_judge_val = sum(1 for l in open(judge_val_path) if l.strip()) - 1  # excl header
                _log(f"[REUSE] judge_val_responses.jsonl already exists ({n_judge_val} rows) — skipping",
                     log_fh)
            else:
                _log("Capturing judge_val_responses ...", log_fh)
                n_judge_val = _capture_judge_responses(
                    client, SERVED_MODEL_NAME, judge_val_examples, judge_val_path,
                    provenance, log_fh, max_tokens=args.max_tokens, stream_name="judge_val",
                )

            # ---- Capture 2: sentinel_responses ----
            if _is_nonempty_capture(sentinel_path):
                n_sentinel = sum(1 for l in open(sentinel_path) if l.strip()) - 1
                _log(f"[REUSE] sentinel_responses.jsonl already exists ({n_sentinel} rows) — skipping",
                     log_fh)
            else:
                _log("Capturing sentinel_responses (24 held-out invalid-PHP prompts) ...", log_fh)
                n_sentinel = _capture_judge_responses(
                    client, SERVED_MODEL_NAME, sentinel_examples, sentinel_path,
                    provenance, log_fh, max_tokens=args.max_tokens, stream_name="sentinel",
                )

            # ---- Capture 3: reasoning_samples ----
            if _is_nonempty_capture(reasoning_path):
                n_reasoning = sum(1 for l in open(reasoning_path) if l.strip()) - 1
                _log(f"[REUSE] reasoning_samples.jsonl exists ({n_reasoning} rows) — skipping",
                     log_fh)
            else:
                _log("Capturing reasoning_samples (cot+ctf) ...", log_fh)
                n_reasoning = _capture_reasoning_samples(
                    client, SERVED_MODEL_NAME, reasoning_examples, reasoning_path,
                    provenance, log_fh, max_tokens=args.max_tokens, stream_name="reasoning",
                )

            # ---- Capture 4: gen_samples ----
            if _is_nonempty_capture(gen_path):
                n_gen = sum(1 for l in open(gen_path) if l.strip()) - 1
                _log(f"[REUSE] gen_samples.jsonl exists ({n_gen} rows) — skipping", log_fh)
            else:
                _log("Capturing gen_samples (ctf) ...", log_fh)
                n_gen = _capture_reasoning_samples(
                    client, SERVED_MODEL_NAME, gen_examples, gen_path,
                    provenance, log_fh, max_tokens=args.max_tokens, stream_name="gen",
                )

            # ---- Capture 5: eval_gen (REVL-02) ----
            if eval_gen_path.exists() and eval_gen_path.stat().st_size > 0:
                _log(f"[REUSE] eval_gen_results.json exists — skipping", log_fh)
                with open(eval_gen_path) as f:
                    gen_result = json.load(f)
                phpcs_pass_rate = gen_result.get("phpcs_pass_rate")
            else:
                _log("Running eval_gen (REVL-02 phpcs_pass_rate) ...", log_fh)
                from eval import eval_gen
                os.environ["EVAL_GEN_BASE_URL"] = endpoint
                gen_result = eval_gen.run_eval(
                    dataset_path=DATASET,
                    limit=None,
                    output_path=str(eval_gen_path),
                    base_url=endpoint,
                )
                phpcs_pass_rate = gen_result.get("phpcs_pass_rate")
                _log(f"eval_gen done: phpcs_pass_rate={phpcs_pass_rate}", log_fh)

        except VllmBootTimeout as e:
            _log(f"ABORT: vLLM boot timeout: {e}", log_fh)
            return 3
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            _log(f"ABORT: unexpected error: {e}", log_fh)
            import traceback
            traceback.print_exc()
            raise
        finally:
            stop_vllm(CONTAINER_NAME)
            _log("vLLM container stopped.", log_fh)

        # ---- Write capture_manifest.json ----
        manifest = {
            "served": served_identity,
            "staging_model_dir": str(staging_dir),
            "enable_thinking": False,
            "port": args.port,
            "gpu_mem_util": args.gpu_mem_util,
            "dataset": DATASET,
            "sentinel_dataset": SENTINEL_DATASET,
            "streams": {
                "judge_val": {
                    "path": str(judge_val_path),
                    "rows": n_judge_val,
                    "schema": "{index, response}",
                },
                "sentinel": {
                    "path": str(sentinel_path),
                    "rows": n_sentinel,
                    "schema": "{index, response}",
                },
                "reasoning_samples": {
                    "path": str(reasoning_path),
                    "rows": n_reasoning,
                    "schema": "{example_idx, task_type, prompt, response, model_scores}",
                },
                "gen_samples": {
                    "path": str(gen_path),
                    "rows": n_gen,
                    "schema": "{example_idx, task_type, prompt, response, model_scores}",
                },
                "eval_gen": {
                    "path": str(eval_gen_path),
                    "phpcs_pass_rate": phpcs_pass_rate,
                },
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        _log(f"capture_manifest.json written: {manifest_path}", log_fh)

        # ---- Summary ----
        _log("=== _04.4_v4winner_serve_capture.py COMPLETE ===", log_fh)
        _log(f"  judge_val rows:       {n_judge_val}", log_fh)
        _log(f"  sentinel rows:        {n_sentinel}", log_fh)
        _log(f"  reasoning rows:       {n_reasoning}", log_fh)
        _log(f"  gen rows:             {n_gen}", log_fh)
        _log(f"  phpcs_pass_rate:      {phpcs_pass_rate}", log_fh)
        _log(f"  served identity:      {served_identity}", log_fh)

        print(f"\n[RESULT] served={served_identity} judge_val={n_judge_val} "
              f"sentinel={n_sentinel} reasoning={n_reasoning} gen={n_gen} "
              f"phpcs_pass_rate={phpcs_pass_rate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
