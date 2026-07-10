#!/usr/bin/env python3
"""Phase 04.4 Plan 04 Task 2 — post-merge 10+10 validation of the PROMOTED canonical v4 model.

Serves models/qwen3-30b-wp-30_70-reasoning-merged-v4, runs 10 <wp_gen> + 10 <wp_judge>
inferences, asserts coherent output + correct task-token routing, writes
output/eval_reasoning_v4_winner/postmerge_validation_v4.json, tears down.

Routing/coherence criteria:
  - wp_judge coherent := response parses to a judge output carrying an overall/dimension score
    (structured rubric verdict) — NOT raw code.
  - wp_gen   coherent := response is a code generation (contains PHP/code markers) — NOT a judge
    JSON verdict. This is the task-token routing check: the same model must answer the two task
    tokens with the two distinct output shapes.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = "models/qwen3-30b-wp-30_70-reasoning-merged-v4"
CONTAINER = "wp-postmerge-validate-vllm"
PORT = 8021
GPU_MEM_UTIL = 0.55
VAL = ROOT / "data/reasoning_dataset/openai_val.jsonl"
OUT = ROOT / "output/eval_reasoning_v4_winner/postmerge_validation_v4.json"
N_EACH = 10


def _user(msgs):
    return next((m["content"] for m in msgs if m.get("role") == "user"), "")


def _is_judge_coherent(text: str) -> bool:
    if not text or not text.strip():
        return False
    # structured rubric verdict: an overall/dimension score present
    return bool(re.search(r'overall_score|wpcs_compliance|security_score', text))


def _is_gen_coherent(text: str) -> bool:
    if not text or not text.strip():
        return False
    # code generation: PHP/code markers and NOT a judge rubric JSON
    if re.search(r'overall_score|wpcs_compliance', text):
        return False
    return bool(re.search(r'<\?php|function\s+\w+\s*\(|add_action|add_filter|\$wpdb|return\s', text))


def main() -> int:
    val = [json.loads(l) for l in open(VAL) if l.strip()]
    judge_rows = [r for r in val if "<wp_judge>" in _user(r.get("messages", []))][:N_EACH]
    gen_rows = [r for r in val if "<wp_gen>" in _user(r.get("messages", []))][:N_EACH]
    print(f"[validate] judge prompts: {len(judge_rows)}  gen prompts: {len(gen_rows)}", flush=True)

    from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm
    import openai

    endpoint = f"http://localhost:{PORT}/v1"
    result = {
        "model": MODEL_DIR, "served_identity": None, "enable_thinking": False,
        "wp_gen_total": len(gen_rows), "wp_judge_total": len(judge_rows),
        "wp_gen_coherent": 0, "wp_judge_coherent": 0,
        "routing_correct": 0, "routing_total": len(gen_rows) + len(judge_rows),
        "samples": [],
    }
    try:
        print(f"[validate] booting {CONTAINER} on :{PORT} model={MODEL_DIR}", flush=True)
        boot_vllm(MODEL_DIR, CONTAINER, PORT, GPU_MEM_UTIL)
        served = wait_healthy(PORT, CONTAINER)
        result["served_identity"] = served
        print(f"[validate] healthy; served={served}", flush=True)
        client = openai.OpenAI(base_url=endpoint, api_key="x")

        def ask(prompt: str) -> str:
            r = client.chat.completions.create(
                model="wp-30_70",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return r.choices[0].message.content or ""

        for r in judge_rows:
            p = _user(r["messages"])
            txt = ask(p)
            ok = _is_judge_coherent(txt)
            routed = ok  # judge token -> judge-shaped output
            result["wp_judge_coherent"] += int(ok)
            result["routing_correct"] += int(routed)
            result["samples"].append({"task": "wp_judge", "coherent": ok, "len": len(txt)})
            print(f"[validate] wp_judge coherent={ok} len={len(txt)}", flush=True)

        for r in gen_rows:
            p = _user(r["messages"])
            txt = ask(p)
            ok = _is_gen_coherent(txt)
            routed = ok  # gen token -> code-shaped output (not a judge JSON)
            result["wp_gen_coherent"] += int(ok)
            result["routing_correct"] += int(routed)
            result["samples"].append({"task": "wp_gen", "coherent": ok, "len": len(txt)})
            print(f"[validate] wp_gen coherent={ok} len={len(txt)}", flush=True)
    finally:
        stop_vllm(CONTAINER)
        print("[validate] container stopped.", flush=True)

    result["pass"] = (result["wp_gen_coherent"] >= 9 and result["wp_judge_coherent"] >= 9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"[validate] DONE wp_gen={result['wp_gen_coherent']}/{result['wp_gen_total']} "
          f"wp_judge={result['wp_judge_coherent']}/{result['wp_judge_total']} "
          f"routing={result['routing_correct']}/{result['routing_total']} pass={result['pass']}", flush=True)
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
