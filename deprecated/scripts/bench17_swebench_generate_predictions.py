#!/usr/bin/env python3
"""BENCH-02 generation-mode prediction generator (Phase 17-03, Task 1).

Generates SWE-bench predictions.jsonl for the pre-registered scope
(output/bench17/swebench_scope_preregistration.md): SWE-bench Lite-300
(oracle) primary + SWE-bench-Multilingual PHP-43 (oracle-equivalent)
secondary. Non-agentic generation-mode: one prompt in, one unified-diff
patch out, via the same served model/config the 17-02 throughput probe
measured (vLLM bf16, max_model_len=24576, concurrency=2, temp 0.0, seed 0,
max_tokens 2048, enable_thinking=false).

Prompt construction:
  - Lite-300: princeton-nlp/SWE-bench_Lite_oracle 'text' field verbatim
    (official pre-built oracle prompt, style-2 format).
  - PHP-43: no pre-built oracle HF variant exists for Multilingual, so this
    builds an oracle-equivalent prompt via swebench's own
    `swebench.inference.make_datasets.create_instance.add_text_inputs`
    (file_source="oracle", prompt_style="style-2") -- the exact official
    prompt-construction pipeline that produced *_oracle datasets upstream,
    NOT a hand-rolled template. This clones the 4 PHP repos locally (once)
    via AutoContextManager.

Patch post-processing follows the official swebench.inference pipeline
(swebench/inference/run_live.py): raw model text -> extract_diff (pulls
content out of <patch>/```diff fences) -> extract_minimal_patch (recomputes
hunk line-count headers). Never hand-rolled.

Resumable: predictions files are opened in append mode; already-written
instance_ids are skipped on restart (safe to re-run after an interruption).

Over-length prompts (> max_model_len - max_tokens) are NOT silently
excluded: the request is submitted anyway, and if the server rejects it
(context-length error) or returns an empty patch, the row is still written
with model_patch="" plus disclosure fields (_over_length/_error), so the
row is later scored unresolved by the harness and counted in the receipt --
per the LOCKED pre-registration handling.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import (  # noqa: E402
    SERVE_SCRIPT,
    VllmBootTimeout,
    wait_healthy,
    stop_vllm,
)

MODEL_DIR = PROJECT_ROOT / "models" / "qwen3-30b-wp-30_70-reasoning-merged-v4"
MODEL_NAME_OR_PATH = "qwen3-30b-wp-30_70-reasoning-merged-v4"
CONTAINER_NAME = "wp-bench17-swebench-gen-vllm"
PORT = 8020
GPU_MEM_UTIL = 0.55
MAX_MODEL_LEN = 24576
MAX_TOKENS = 2048
CONCURRENCY = 2  # matches the 17-02 throughput probe / pre-registration
SEED = 0
OUT_DIR = PROJECT_ROOT / "output" / "bench17"

PHP_REPOS = {
    "phpoffice/phpspreadsheet",
    "laravel/framework",
    "php-cs-fixer/php-cs-fixer",
    "briannesbitt/carbon",
}

write_lock = threading.Lock()


def boot_vllm_wide_ctx() -> None:
    env = {
        **os.environ,
        "CONTAINER_NAME": CONTAINER_NAME,
        "PORT": str(PORT),
        "MODEL_DIR": str(MODEL_DIR),
        "GPU_MEM_UTIL": str(GPU_MEM_UTIL),
        "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
    }
    print(f"[vllm] booting {CONTAINER_NAME} on :{PORT} max_model_len={MAX_MODEL_LEN}", flush=True)
    subprocess.run(
        ["bash", SERVE_SCRIPT], env=env, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def build_php43_text_inputs() -> dict[str, str]:
    """Oracle-equivalent prompts for the PHP-Multilingual subset via the
    official swebench make_datasets pipeline (not hand-rolled)."""
    from datasets import load_dataset
    from swebench.inference.make_datasets.create_instance import add_text_inputs

    progress_file = OUT_DIR / "php43_text_inputs.jsonl"
    print("[php43] loading SWE-bench/SWE-bench_Multilingual test split...", flush=True)
    ds = load_dataset("SWE-bench/SWE-bench_Multilingual", split="test")
    php_rows = [dict(row) for row in ds if row["repo"] in PHP_REPOS]
    assert len(php_rows) == 43, f"expected 43 PHP instances, got {len(php_rows)}"
    instances = {row["instance_id"]: row for row in php_rows}

    add_text_inputs(
        instances,
        retrieval_file=None,
        k=None,
        prompt_style="style-2",
        file_source="oracle",
        progress_file=str(progress_file),
    )

    text_by_id: dict[str, str] = {}
    with open(progress_file) as f:
        for line in f:
            row = json.loads(line)
            if row.get("text_inputs"):
                text_by_id[row["instance_id"]] = row["text_inputs"]
    missing = set(instances) - set(text_by_id)
    if missing:
        print(f"[php43] WARNING: {len(missing)} instances failed prompt construction: {sorted(missing)}", flush=True)
    print(f"[php43] built {len(text_by_id)}/{len(instances)} oracle-equivalent prompts", flush=True)
    return text_by_id


def load_lite300_text_inputs() -> dict[str, str]:
    from datasets import load_dataset

    print("[lite300] loading princeton-nlp/SWE-bench_Lite_oracle test split...", flush=True)
    ds = load_dataset("princeton-nlp/SWE-bench_Lite_oracle", split="test")
    text_by_id = {row["instance_id"]: row["text"] for row in ds}
    assert len(text_by_id) == 300, f"expected 300 Lite instances, got {len(text_by_id)}"
    print(f"[lite300] loaded {len(text_by_id)} prompts", flush=True)
    return text_by_id


def already_done(pred_path: Path) -> set[str]:
    if not pred_path.exists():
        return set()
    done = set()
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["instance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def generate_one(client, served_model: str, instance_id: str, text: str) -> dict:
    from swebench.inference.make_datasets.utils import extract_diff, extract_minimal_patch

    row = {
        "instance_id": instance_id,
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "model_patch": "",
        "_over_length": False,
        "_error": None,
    }
    try:
        resp = client.chat.completions.create(
            model=served_model,
            messages=[{"role": "user", "content": text}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            seed=SEED,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
        diff = extract_diff(raw)
        patch = extract_minimal_patch(diff) if diff else ""
        row["model_patch"] = patch or ""
        if not patch:
            row["_error"] = "empty_or_unparseable_patch"
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        row["_error"] = msg[:500]
        if "maximum context length" in msg.lower() or "context_length" in msg.lower() or "400" in msg[:10]:
            row["_over_length"] = True
    return row


def run_variant(client, served_model: str, text_by_id: dict[str, str], pred_path: Path, label: str) -> dict:
    done = already_done(pred_path)
    todo = {iid: t for iid, t in text_by_id.items() if iid not in done}
    print(f"[{label}] {len(done)} already done, {len(todo)} to generate (of {len(text_by_id)} scoped)", flush=True)
    n_over_length = 0
    n_error = 0
    n_ok = 0
    t0 = time.time()
    with open(pred_path, "a") as f:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {
                pool.submit(generate_one, client, served_model, iid, t): iid
                for iid, t in todo.items()
            }
            n_completed = 0
            for fut in as_completed(futures):
                row = fut.result()
                n_completed += 1
                if row["_over_length"]:
                    n_over_length += 1
                elif row["_error"]:
                    n_error += 1
                else:
                    n_ok += 1
                with write_lock:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                if n_completed % 10 == 0 or n_completed == len(todo):
                    elapsed = time.time() - t0
                    print(f"[{label}] {n_completed}/{len(todo)} done ({elapsed:.0f}s elapsed, "
                          f"ok={n_ok} over_length={n_over_length} error={n_error})", flush=True)
    wall = time.time() - t0
    return {
        "label": label,
        "instances_scoped": len(text_by_id),
        "instances_already_done_at_start": len(done),
        "instances_attempted_this_run": len(todo),
        "ok": n_ok,
        "over_length": n_over_length,
        "error_non_over_length": n_error,
        "wall_clock_s_this_run": round(wall, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-php", action="store_true", help="skip the PHP-43 secondary variant")
    ap.add_argument("--skip-lite", action="store_true", help="skip the Lite-300 primary variant")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lite_path = OUT_DIR / "swebench_predictions.jsonl"
    php_path = OUT_DIR / "swebench_predictions_php.jsonl"
    receipt_path = OUT_DIR / "swebench_generation_receipt.json"

    text_by_id_lite = {} if args.skip_lite else load_lite300_text_inputs()
    text_by_id_php = {} if args.skip_php else build_php43_text_inputs()

    import openai

    t0 = time.time()
    served = None
    variant_receipts = []
    try:
        boot_vllm_wide_ctx()
        served = wait_healthy(PORT, CONTAINER_NAME)
        client = openai.OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="none")

        warm = client.chat.completions.create(
            model=served,
            messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
            max_tokens=16,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        warm_text = (warm.choices[0].message.content or "").strip()
        if not warm_text:
            raise RuntimeError(f"Real-generation warm-up returned empty output: {warm!r}")
        print(f"[warmup] real-generation OK (served_model={served!r}): {warm_text[:80]!r}", flush=True)

        if text_by_id_lite:
            variant_receipts.append(run_variant(client, served, text_by_id_lite, lite_path, "lite300"))
        if text_by_id_php:
            variant_receipts.append(run_variant(client, served, text_by_id_php, php_path, "php43"))
    except VllmBootTimeout as e:
        print(f"HALT: vLLM boot timeout: {e}", file=sys.stderr)
        return 3
    finally:
        stop_vllm(CONTAINER_NAME)

    total_wall = time.time() - t0
    receipt = {
        "task": "swebench_generate_predictions",
        "model_dir": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "served_model_name": served,
        "serving_config": {
            "engine": "vLLM", "dtype": "bf16", "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEM_UTIL, "concurrency": CONCURRENCY, "port": PORT,
        },
        "sampling_config": {
            "temperature": 0.0, "max_tokens": MAX_TOKENS, "seed": SEED,
            "enable_thinking": False, "retrieval_style": "oracle",
        },
        "prompt_construction": {
            "lite300": "princeton-nlp/SWE-bench_Lite_oracle 'text' field verbatim (official style-2 oracle prompt)",
            "php43": "swebench.inference.make_datasets.create_instance.add_text_inputs(file_source='oracle', prompt_style='style-2') -- official pipeline, oracle-equivalent",
        },
        "patch_postprocessing": "extract_diff -> extract_minimal_patch (swebench.inference.make_datasets.utils, official pipeline, run_live.py order)",
        "warmup": {"gated_on_real_generation": True},
        "variants": variant_receipts,
        "total_wall_clock_s": round(total_wall, 1),
        "total_wall_clock_h": round(total_wall / 3600, 2),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print(f"\nWritten: {receipt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
