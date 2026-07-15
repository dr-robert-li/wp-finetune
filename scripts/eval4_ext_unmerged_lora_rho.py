#!/usr/bin/env python
"""Phase 23-03 extension -- serve the v4 judge adapter UNMERGED (runtime
LoRA) via vLLM --enable-lora, using the converted PEFT-convention adapter
(scripts/eval4_ext_unmerged_lora_convert.py), and score against the same
121 wp_judge val prompts as every other v4 judge measurement.

Re-derives the boot/diff-gate/capture pattern from
scripts/build_exp21_unmerged_lora_rho.py (that experiment blocked on
Tinker's raw w1/w2/w3 naming; this one uses the renamed/reshaped adapter
that matches vLLM's FusedMoE3DWithLoRA PEFT convention -- see
output/eval4/ext_unmerged_preregistration.md for the derivation).

Usage:
  python scripts/eval4_ext_unmerged_lora_rho.py --seed s1
  python scripts/eval4_ext_unmerged_lora_rho.py --seed s0
  python scripts/eval4_ext_unmerged_lora_rho.py --seed s2
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm, VllmBootTimeout  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "eval4"
UNMERGED_DIR = OUT_DIR / "ext_unmerged"
BASE_MODEL_DIR = "models/Qwen3.6-35B-A3B"
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
PORT = 8026
GPU_MEM_UTIL = 0.80
MAX_MODEL_LEN = 16384
MAX_TOKENS = 8192
BOOT_TIMEOUT_SEC = 1200
DATASET = "data/reasoning_dataset/openai_val.jsonl"
DIFF_PROMPT = "<wp_judge> Evaluate this WordPress code:\n\n```php\nfunction f() { return 1; }\n```"
# Multiple diff prompts, not one: a single-prompt greedy-decode diff gate proved flaky across
# repeated identical-config boots (observed 2026-07-15 -- run 1 differed, run 2 did not, on the
# SAME prompt/adapter/config), most likely due to first-hit Triton JIT warmup / MoE routing
# nondeterminism in this serving stack rather than a real load failure. Requiring "differs on at
# least 1 of N distinct prompts" is far more robust than a single coincidental match/non-match.
DIFF_PROMPTS = [
    DIFF_PROMPT,
    "<wp_judge> Evaluate this WordPress code:\n\n```php\n$wpdb->query(\"SELECT * FROM wp_posts WHERE id = \" . $_GET['id']);\n```",
    "<wp_judge> Evaluate this WordPress code:\n\n```php\nfunction wpcs_get_option( $key ) {\n\treturn get_option( $key, false );\n}\n```",
]

# Anchors (VERDICT-EVAL4.md / judge03_rho.json / judge03_capture_rho.json)
CAPTURE_ANCHOR_RHO = 0.8358149892119933  # s1 Tinker capture, best single seed
CAPTURE_ANCHOR_CI = (0.7740, 0.8526287320467012)  # s1 capture CI (see ext_q8_results.json v4_ensemble CI upper as proxy upper if needed)
SERVED_MERGED_ANCHOR_RHO = 0.7872  # s1 vLLM-served-merged


def _src_adapter_dir(seed: str) -> Path:
    return PROJECT_ROOT / "output" / "base21" / f"judge03_{seed}_adapter"


def _converted_adapter_dir(seed: str) -> Path:
    return UNMERGED_DIR / f"judge03_{seed}_adapter_vllm_peft"


def _ensure_converted(seed: str) -> Path:
    dst = _converted_adapter_dir(seed)
    if (dst / "adapter_model.safetensors").exists():
        return dst
    from scripts.eval4_ext_unmerged_lora_convert import convert, validate  # noqa: PLC0415

    src = _src_adapter_dir(seed)
    if not (src / "adapter_model.safetensors").exists():
        raise RuntimeError(
            f"source adapter not found for seed={seed}: {src} -- download it first "
            f"(output/tinker/wp-judge-v4-{seed}-manifest.json has the tinker:// sampler_path)"
        )
    receipt = convert(src, dst)
    check = validate(dst)
    receipt["validation"] = check
    (OUT_DIR / f"ext_unmerged_convert_receipt_{seed}.json").write_text(json.dumps(receipt, indent=2))
    if not check["coverage_ok"]:
        raise RuntimeError(f"conversion validation failed for seed={seed}: {check}")
    return dst


def _boot(adapter_dir: Path, container: str, lora_name: str) -> dict:
    try:
        boot_vllm(BASE_MODEL_DIR, container, PORT, GPU_MEM_UTIL,
                  serve_script=SERVE_SCRIPT,
                  extra_env={"LANGUAGE_MODEL_ONLY": "1", "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
                             "LORA_ADAPTER_DIR": str(adapter_dir.resolve()),
                             "LORA_NAME": lora_name, "LORA_MAX_RANK": "32"})
    except RuntimeError as e:
        return {"boot_ok": False, "stage": "docker_run", "error": str(e)}
    try:
        wait_healthy(PORT, container, timeout=BOOT_TIMEOUT_SEC)
    except VllmBootTimeout as e:
        return {"boot_ok": False, "stage": "wait_healthy", "error": str(e)}
    return {"boot_ok": True}


def _diff_gate(lora_name: str) -> dict:
    per_prompt = []
    any_differs = False
    any_nonempty = False
    for i, prompt in enumerate(DIFF_PROMPTS):
        lora_out = generate(PORT, lora_name, [{"instruction": prompt, "source_val_idx": f"diff{i}"}], max_tokens=256)
        base_out = generate(PORT, "/workspace/model", [{"instruction": prompt, "source_val_idx": f"diff{i}"}], max_tokens=256)
        lora_text = (lora_out[0] or "").strip()
        base_text = (base_out[0] or "").strip()
        differs = bool(lora_text) and lora_text != base_text
        any_differs = any_differs or differs
        any_nonempty = any_nonempty or bool(lora_text)
        per_prompt.append({
            "lora_output": lora_text[:300],
            "base_output": base_text[:300],
            "lora_output_empty": not lora_text,
            "differs_from_base": differs,
        })
    return {
        "per_prompt": per_prompt,
        "lora_output_empty": not any_nonempty,
        "differs_from_base": any_differs,
    }


def _capture_and_score(seed: str, lora_name: str) -> dict:
    from scripts.sieve_capture_judge_http import capture as http_capture  # noqa: PLC0415

    cap_path = UNMERGED_DIR / f"capture_unmerged_lora_{seed}.jsonl"
    warm = generate(PORT, lora_name, [{"instruction": "Reply with exactly one word: OK", "source_val_idx": "warmup"}], max_tokens=16)
    if not warm or not warm[0].strip():
        raise RuntimeError(f"real-generation warm-up returned empty output: {warm!r}")
    print(f"[ext-unmerged {seed}] warm-up OK: {warm[0].strip()[:80]!r}", flush=True)

    cap_stats = http_capture(base_url=f"http://localhost:{PORT}/v1", model=lora_name,
                              dataset=DATASET, out=str(cap_path), max_tokens=MAX_TOKENS, temperature=0.0)
    print(f"[ext-unmerged {seed}] capture stats: {cap_stats}", flush=True)

    r = subprocess.run([sys.executable, "scripts/relabel/eval_relabel.py", str(cap_path)],
                        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"eval_relabel failed on unmerged-LoRA capture (exit {r.returncode})")
    summary = json.loads((PROJECT_ROOT / "output" / "eval_summary.json").read_text())
    m = re.search(r"parse_fail=(\d+)", r.stdout)
    parse_fail = int(m.group(1)) if m else summary.get("parse_fail")

    return {
        "rho": summary["rho_new"],
        "ci_lower": summary["ci"][0],
        "ci_upper": summary["ci"][1],
        "n": summary["n"],
        "parse_fail": parse_fail,
        "capture_path": str(cap_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, choices=["s0", "s1", "s2"])
    args = ap.parse_args()
    seed = args.seed

    UNMERGED_DIR.mkdir(parents=True, exist_ok=True)
    container = f"ext-unmerged-{seed}"
    lora_name = f"judge-{seed}"
    out_path = OUT_DIR / f"ext_unmerged_lora_rho_{seed}.json"
    t0 = time.time()

    result: dict = {"experiment": f"ext_unmerged_lora_vllm_{seed}", "base_model_dir": BASE_MODEL_DIR,
                     "max_tokens": MAX_TOKENS, "max_model_len_served": MAX_MODEL_LEN, "temperature": 0.0}

    try:
        adapter_dir = _ensure_converted(seed)
        result["adapter_dir"] = str(adapter_dir)

        boot = _boot(adapter_dir, container, lora_name)
        result["boot"] = boot
        if not boot["boot_ok"]:
            result["status"] = "blocked"
            result["blocked_reason"] = (
                f"vLLM failed to boot/load the converted PEFT-convention routed-MoE-expert LoRA "
                f"adapter via --enable-lora (stage={boot['stage']}) -- the renamed/reshaped "
                f"adapter (base_layer/experts convention) was still rejected at a deeper level "
                f"than naming. Falls through to the llama.cpp fallback per pre-registration."
            )
            result["wall_clock_s"] = round(time.time() - t0, 1)
            out_path.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
            return 0

        diff = _diff_gate(lora_name)
        result["diff_gate"] = diff
        if diff["lora_output_empty"] or not diff["differs_from_base"]:
            result["status"] = "blocked"
            result["blocked_reason"] = (
                "vLLM booted with --enable-lora but the real-generation diff gate shows the "
                "LoRA adapter had NO measurable effect vs the raw base -- refusing to spend a "
                "121-item capture on a degraded/no-op adapter load."
            )
            result["wall_clock_s"] = round(time.time() - t0, 1)
            out_path.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
            return 0

        scored = _capture_and_score(seed, lora_name)
        result["unmerged_lora"] = scored
        result["status"] = "measured"
        if seed == "s1":
            rho = scored["rho"]
            d_capture = abs(rho - CAPTURE_ANCHOR_RHO)
            d_served_merged = abs(rho - SERVED_MERGED_ANCHOR_RHO)
            result["h1_check"] = {
                "capture_anchor_rho": CAPTURE_ANCHOR_RHO,
                "served_merged_anchor_rho": SERVED_MERGED_ANCHOR_RHO,
                "rho_measured": rho,
                "delta_vs_capture_anchor": round(rho - CAPTURE_ANCHOR_RHO, 4),
                "delta_vs_served_merged_anchor": round(rho - SERVED_MERGED_ANCHOR_RHO, 4),
                "closer_to_capture": d_capture < d_served_merged,
                "h1_confirmed": d_capture < d_served_merged and rho >= (CAPTURE_ANCHOR_RHO - 0.03),
            }
    except Exception as exc:  # noqa: BLE001 -- fail-closed receipt even on unexpected errors
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        stop_vllm(container)

    result["wall_clock_s"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[ext-unmerged {seed}] wrote {out_path}", flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
