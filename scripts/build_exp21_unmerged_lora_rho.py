#!/usr/bin/env python
"""Phase 21 Diagnostic Experiment 2 -- JUDGE-03 unmerged-LoRA-via-vLLM
confirming experiment (judge_attenuation_forensics.md Sec 5, DIAGNOSTIC_SYNTHESIS.md #2).

Serves the RAW base model + the promoted s1 judge LoRA adapter natively via
vLLM --enable-lora (no merge step at all), captures the identical 121
wp_judge val prompts, scores with the unmodified eval_relabel.py, and
compares rho against the two existing anchors:
  - 0.8358 (Tinker-capture, same checkpoint)  -> merge-numerics hypothesis
  - 0.7872 (vLLM-served MERGED, same checkpoint) -> engine-numerics hypothesis

CAVEAT (pre-registered, honest-blocked-not-degraded): the s1 adapter is a
routed-MoE-expert LoRA -- 120/240 modules are `mlp.experts.{w1,w2,w3}`
(PEFT `target_parameters` 3D per-expert tensors), the other 120 are ordinary
`mlp.shared_expert.{gate,up,down}_proj` `target_modules` nn.Linear tensors.
vLLM's --enable-lora path is built for target_modules; whether it can even
load the routed-expert half is UNVERIFIED going in -- this experiment tests
that directly via a real-generation diff gate BEFORE spending the full
121-prompt capture. If vLLM fails to boot/load the adapter, or the diff gate
shows the adapter had literally zero effect vs the raw base, this is
recorded as status=blocked with evidence -- not silently downgraded to an
attn-only/shared-expert-only partial test that would answer a different
question than the one asked.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm, VllmBootTimeout  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "base21"
DIAG_DIR = OUT_DIR / "diagnostic"
BASE_MODEL_DIR = "models/Qwen3.6-35B-A3B"
ADAPTER_DIR = str(OUT_DIR / "judge03_s1_adapter")
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
CONTAINER = "exp2-unmerged-lora"
PORT = 8025
GPU_MEM_UTIL = 0.80
MAX_MODEL_LEN = 16384  # same as 21-06 (2288-token longest wp_judge prompt + 8192 completion cap)
MAX_TOKENS = 8192
BOOT_TIMEOUT_SEC = 1200  # 67 GiB base, Pitfall 3 lesson
LORA_NAME = "judge-s1"
DATASET = "data/reasoning_dataset/openai_val.jsonl"
DIFF_PROMPT = "<wp_judge> Evaluate this WordPress code:\n\n```php\nfunction f() { return 1; }\n```"

CAPTURE_ANCHOR_RHO = 0.8358149892119933  # judge03_capture_rho.json best_single_seed (Tinker capture)
SERVED_MERGED_ANCHOR_RHO = 0.7872  # judge03_rho.json vllm_served_single_seed (merged)
RHO_PROXIMITY_BAND = 0.02  # within this band of an anchor => that hypothesis wins outright


def _adapter_module_breakdown() -> dict:
    from safetensors import safe_open
    import re

    with safe_open(str(Path(ADAPTER_DIR) / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    mods = {re.sub(r"\.lora_[AB].*$", "", k) for k in keys}
    routed = sorted(m for m in mods if ".mlp.experts." in m)
    other = sorted(m for m in mods if ".mlp.experts." not in m)
    return {"total_modules": len(mods), "routed_expert_modules": len(routed), "shared_expert_modules": len(other)}


def _boot() -> dict:
    """Boot vLLM with --enable-lora, returning boot evidence. Raises nothing;
    boot failure is captured as a dict (status=blocked) so the caller can
    write a fail-closed receipt without a bare exception."""
    try:
        boot_vllm(BASE_MODEL_DIR, CONTAINER, PORT, GPU_MEM_UTIL,
                  serve_script=SERVE_SCRIPT,
                  extra_env={"LANGUAGE_MODEL_ONLY": "1", "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
                             "LORA_ADAPTER_DIR": str(Path(ADAPTER_DIR).resolve()),
                             "LORA_NAME": LORA_NAME, "LORA_MAX_RANK": "32"})
    except RuntimeError as e:
        return {"boot_ok": False, "stage": "docker_run", "error": str(e)}
    try:
        wait_healthy(PORT, CONTAINER, timeout=BOOT_TIMEOUT_SEC)
    except VllmBootTimeout as e:
        return {"boot_ok": False, "stage": "wait_healthy", "error": str(e)}
    return {"boot_ok": True}


def _diff_gate() -> dict:
    """Real-generation diff: LoRA-served vs raw-base-served on the SAME
    running container (vLLM serves both model ids from one endpoint when
    --enable-lora + --lora-modules are set). If identical, the adapter had
    no measurable effect -- do not proceed to a meaningless 121-item capture."""
    lora_out = generate(PORT, LORA_NAME, [{"instruction": DIFF_PROMPT, "source_val_idx": "exp2_diff"}],
                         max_tokens=256)
    base_out = generate(PORT, "/workspace/model", [{"instruction": DIFF_PROMPT, "source_val_idx": "exp2_diff"}],
                         max_tokens=256)
    lora_text = (lora_out[0] or "").strip()
    base_text = (base_out[0] or "").strip()
    return {
        "lora_output": lora_text[:500],
        "base_output": base_text[:500],
        "lora_output_empty": not lora_text,
        "differs_from_base": bool(lora_text) and lora_text != base_text,
    }


def _capture_and_score() -> dict:
    from scripts.sieve_capture_judge_http import capture as http_capture

    cap_path = OUT_DIR / "exp2_judge_capture_unmerged_lora_s1.jsonl"
    warm = generate(PORT, LORA_NAME, [{"instruction": "Reply with exactly one word: OK", "source_val_idx": "warmup"}],
                     max_tokens=16)
    if not warm or not warm[0].strip():
        raise RuntimeError(f"real-generation warm-up (LoRA model id) returned empty output: {warm!r}")
    print(f"[exp2] warm-up OK: {warm[0].strip()[:80]!r}", flush=True)

    cap_stats = http_capture(base_url=f"http://localhost:{PORT}/v1", model=LORA_NAME,
                              dataset=DATASET, out=str(cap_path),
                              max_tokens=MAX_TOKENS, temperature=0.0)
    print(f"[exp2] capture stats: {cap_stats}", flush=True)

    r = subprocess.run([sys.executable, "scripts/relabel/eval_relabel.py", str(cap_path)],
                        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"eval_relabel failed on unmerged-LoRA capture (exit {r.returncode})")
    summary = json.loads((OUT_DIR / "eval_summary.json").read_text())
    import re
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


def _decision(rho: float) -> dict:
    d_capture = abs(rho - CAPTURE_ANCHOR_RHO)
    d_served_merged = abs(rho - SERVED_MERGED_ANCHOR_RHO)
    if d_capture <= RHO_PROXIMITY_BAND and d_capture < d_served_merged:
        verdict = "merge-numerics CONFIRMED"
    elif d_served_merged <= RHO_PROXIMITY_BAND and d_served_merged < d_capture:
        verdict = "engine-numerics"
    else:
        verdict = "mixed"
    return {
        "verdict": verdict,
        "rho_measured": rho,
        "capture_anchor_rho": CAPTURE_ANCHOR_RHO,
        "served_merged_anchor_rho": SERVED_MERGED_ANCHOR_RHO,
        "delta_vs_capture_anchor": round(rho - CAPTURE_ANCHOR_RHO, 4),
        "delta_vs_served_merged_anchor": round(rho - SERVED_MERGED_ANCHOR_RHO, 4),
        "proximity_band": RHO_PROXIMITY_BAND,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAG_DIR / "exp2_unmerged_lora_rho.json"
    t0 = time.time()

    module_breakdown = _adapter_module_breakdown()
    print(f"[exp2] adapter module breakdown: {module_breakdown}", flush=True)

    result: dict = {
        "experiment": "exp2_unmerged_lora_vllm",
        "adapter_dir": ADAPTER_DIR,
        "adapter_module_breakdown": module_breakdown,
        "base_model_dir": BASE_MODEL_DIR,
        "max_tokens": MAX_TOKENS,
        "max_model_len_served": MAX_MODEL_LEN,
        "temperature": 0.0,
    }

    try:
        boot = _boot()
        result["boot"] = boot
        if not boot["boot_ok"]:
            result["status"] = "blocked"
            result["blocked_reason"] = (
                f"vLLM failed to boot/load the routed-MoE-expert LoRA adapter via --enable-lora "
                f"(stage={boot['stage']}) -- this IS a finding (vLLM LoRA on this fused-MoE "
                f"architecture is unsupported), not a bug to route around."
            )
            result["wall_clock_s"] = round(time.time() - t0, 1)
            out_path.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
            return 0

        diff = _diff_gate()
        result["diff_gate"] = diff
        if diff["lora_output_empty"] or not diff["differs_from_base"]:
            result["status"] = "blocked"
            result["blocked_reason"] = (
                "vLLM booted with --enable-lora but the real-generation diff gate shows the "
                "LoRA adapter had NO measurable effect vs the raw base (lora_output_empty="
                f"{diff['lora_output_empty']}, differs_from_base={diff['differs_from_base']}) -- "
                "most likely vLLM silently loaded only the shared_expert (target_modules) half "
                "and dropped/ignored the 120 routed-expert (target_parameters) modules, or "
                "ignored the adapter entirely. Refusing to spend a 121-item capture on a "
                "degraded/no-op adapter load that would test a different thing than requested."
            )
            result["wall_clock_s"] = round(time.time() - t0, 1)
            out_path.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
            return 0

        vllm_lora = _capture_and_score()
        result["vllm_unmerged_lora"] = vllm_lora
        result["decision"] = _decision(vllm_lora["rho"])
        result["status"] = "measured"
        result["caveat"] = (
            "vLLM booted successfully and the diff gate confirmed the adapter has SOME effect "
            "vs raw base; this does not by itself prove the FULL 240-module adapter (both routed "
            "AND shared-expert halves) was applied identically to the merge path -- see "
            "adapter_module_breakdown + diff_gate for the evidence available to judge that."
        )
    except Exception as exc:  # noqa: BLE001 -- fail-closed receipt even on unexpected errors
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        stop_vllm(CONTAINER)

    result["wall_clock_s"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[exp2] wrote {out_path}", flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
