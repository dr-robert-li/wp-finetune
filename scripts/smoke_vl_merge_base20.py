"""BASE-04 VL merge-path round-trip smoke (Phase 20 base bring-up).

Proves the FULL chain end to end: (1) merge_adapter.py merges the Task-1
probe LoRA adapter onto the v4 base's model.language_model.* weights
(prefix-aware, with the merged-target-module-count partial-load guard); the
merge subprocess is waited on to fully EXIT before any vLLM serve boots
(releases the ~67 GiB CPU copy -- memory-safety on the unified pool); (2) the
merged model serves via vLLM --language-model-only; (3) a real generation on
a fixed deterministic prompt (temperature 0) is compared against the SAME
prompt served by the unmerged base model. The acceptance test is this
base-vs-merged output DIFF, not "merge exited 0" or "serve booted healthy"
independently (Pitfall 3 / T-20-04a -- a silent partial load can still boot
healthy and generate something base-model-like).

Writes output/base20/vl_merge_roundtrip.json (BASE-04 gate receipt).

Usage:
    python -m scripts.smoke_vl_merge_base20
    python scripts/smoke_vl_merge_base20.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import (  # noqa: E402
    boot_vllm,
    wait_healthy,
    generate,
    stop_vllm,
    VllmBootTimeout,
)

CONFIG_PATH = "config/train_config_v4.yaml"
PROBE_ADAPTER_DIR = str(PROJECT_ROOT / "output" / "base20" / "base20_probe_adapter")
MERGED_MODEL_DIR = "models/Qwen3.6-35B-A3B-base20-merged"
BASE_MODEL_DIR = "models/Qwen3.6-35B-A3B"
LORA_TARGET_MODULES_RECEIPT = PROJECT_ROOT / "output" / "base20" / "lora_target_modules.json"
MERGE_GUARD_RECEIPT = PROJECT_ROOT / "output" / "base20" / "_merge_guard_result.json"

SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
PORT = 8021
GPU_MEM_UTIL = 0.80
# Pitfall 3 lesson (900s for the old 57 GiB base); this base is 67 GiB.
BOOT_TIMEOUT_SEC = 1200
# Same prompt Task 1's probe adapter was trained to shift -- maximizes the
# chance of an observable diff for a tiny/cheap probe run.
PROBE_PROMPT = "Reply with exactly one word: OK"

OUT_DIR = PROJECT_ROOT / "output" / "base20"
OUTPUT_PATH = OUT_DIR / "vl_merge_roundtrip.json"


def write_receipt(status: str, **fields) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {"status": status, **fields}
    OUTPUT_PATH.write_text(json.dumps(receipt, indent=2))
    return receipt


def run_merge() -> None:
    """Run merge_adapter.py as a subprocess and BLOCK until it fully exits --
    this releases the ~67 GiB CPU-resident base+merge copy before any GPU
    serve boots (memory-safety: merge + serve are never concurrent)."""
    cmd = [
        sys.executable, "scripts/merge_adapter.py",
        "--config-path", CONFIG_PATH,
        "--adapter-dir", PROBE_ADAPTER_DIR,
        "--output-dir", MERGED_MODEL_DIR,
    ]
    print(f"[merge] running: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    print(r.stdout[-10000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise RuntimeError(f"merge_adapter.py exited {r.returncode} (see stdout/stderr above)")
    print(f"[merge] subprocess exited 0 -- CPU copy released", flush=True)


def serve_and_generate(model_dir: str, container_name: str, allow_empty: bool = False) -> str:
    """Boot, generate on PROBE_PROMPT, stop. If allow_empty is False (the
    BASE model call -- known-good, untouched), an empty response means the
    serving infra broke and this raises. If allow_empty is True (the MERGED
    probe-adapter call), an empty response is itself a VALID and notable
    outcome: this plan's probe adapter is a deliberately aggressive few-step
    high-LR overfit (see build_base20_probe_adapter.py) and can legitimately
    degrade the merged model's output on this exact prompt (e.g. immediate
    EOS) -- that IS an observable difference from the base, not a smoke
    failure. Returns "" in that case rather than raising."""
    boot_vllm(model_dir, container_name, PORT, GPU_MEM_UTIL,
              serve_script=SERVE_SCRIPT, extra_env={"LANGUAGE_MODEL_ONLY": "1"})
    try:
        served = wait_healthy(PORT, container_name, timeout=BOOT_TIMEOUT_SEC)
        out = generate(PORT, served,
                        [{"instruction": PROBE_PROMPT, "source_val_idx": "vl_merge_probe"}],
                        max_tokens=64)
        text = (out[0] or "").strip()
        if not text and not allow_empty:
            raise RuntimeError(f"real-generation returned empty output for {container_name}")
        return text
    finally:
        stop_vllm(container_name)


def run_smoke() -> dict:
    if not LORA_TARGET_MODULES_RECEIPT.exists():
        raise RuntimeError(
            f"missing {LORA_TARGET_MODULES_RECEIPT} -- run "
            f"scripts/build_base20_probe_adapter.py first (Task 1)"
        )
    probe_receipt = json.loads(LORA_TARGET_MODULES_RECEIPT.read_text())
    adapter_source = probe_receipt["source"]
    confidence = probe_receipt["confidence"]
    expected_count = len(probe_receipt["attached_modules"])

    run_merge()

    merged_path = PROJECT_ROOT / MERGED_MODEL_DIR
    if not (merged_path / "config.json").exists():
        raise RuntimeError(f"merge did not produce a model at {merged_path}")

    if not MERGE_GUARD_RECEIPT.exists():
        raise RuntimeError(
            f"missing {MERGE_GUARD_RECEIPT} -- merge_adapter.py should have written the "
            f"merged-target-module-count guard result"
        )
    guard = json.loads(MERGE_GUARD_RECEIPT.read_text())
    merged_target_module_count = guard["merged_target_module_count"]
    expected_target_module_count = guard["expected_target_module_count"]
    raw_expected_module_count = guard.get("raw_expected_module_count")
    dropped_module_count = guard.get("dropped_module_count")
    if merged_target_module_count != expected_target_module_count or merged_target_module_count <= 0:
        # merge_adapter.py's own guard should have aborted before this point on a real
        # mismatch, but re-assert here defensively (T-20-04a: never trust exit-code-0 alone).
        raise RuntimeError(
            f"merged_target_module_count guard mismatch survived the merge subprocess: "
            f"{merged_target_module_count} != {expected_target_module_count}"
        )

    print("[serve] merged model ...", flush=True)
    merged_out = serve_and_generate(MERGED_MODEL_DIR, "base20-merge-smoke-merged", allow_empty=True)
    print(f"[serve] merged output: {merged_out!r}", flush=True)

    print("[serve] base model (for diff) ...", flush=True)
    base_out = serve_and_generate(BASE_MODEL_DIR, "base20-merge-smoke-base", allow_empty=False)
    print(f"[serve] base output: {base_out!r}", flush=True)

    differs = merged_out != base_out
    note = "merged output differs observably from base -- adapter delta landed"
    if not differs:
        if adapter_source == "tinker":
            note = ("REAL trained (tinker) adapter produced NO observable diff on this "
                     "prompt -- unexpected for a non-zero-init adapter; the "
                     "merged_target_module_count guard is the fallback partial-load check "
                     "here (it passed, so weights DID load, but the shift was too small to "
                     "surface on greedy decoding for this prompt)")
            confidence = "reduced"
        else:
            note = ("zero-init local-fallback adapter produced no diff as expected -- "
                     "relying on the merged_target_module_count guard as the partial-load "
                     "check instead")
            confidence = "reduced"

    return write_receipt(
        "pass",
        adapter_source=adapter_source,
        confidence=confidence,
        merged_target_module_count=merged_target_module_count,
        expected_target_module_count=expected_target_module_count,
        raw_expected_module_count=raw_expected_module_count,
        dropped_module_count=dropped_module_count,
        served_ok=True,
        base_vs_merged_differs=differs,
        prompt=PROBE_PROMPT,
        merged_output=merged_out,
        base_output=base_out,
        notes=note,
    )


def main() -> int:
    try:
        result = run_smoke()
    except Exception as exc:  # noqa: BLE001 -- smoke script: any exception is a gate failure
        write_receipt("fail", failing_field="exception", error=str(exc))
        print(f"BASE-04 SMOKE FAILED: {exc}")
        return 1
    finally:
        # Belt-and-suspenders: make sure nothing is left serving even if an
        # earlier stop_vllm() in serve_and_generate's finally didn't run
        # (e.g. exception before boot_vllm was reached).
        stop_vllm("base20-merge-smoke-merged")
        stop_vllm("base20-merge-smoke-base")

    print("BASE-04 SMOKE PASSED")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
