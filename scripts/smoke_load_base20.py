"""Load-smoke test for the Qwen3.6-35B-A3B v4 base (BASE-01 gate).

Loads the base model via transformers with trust_remote_code=True, asserts the
resolved architecture class, runs a single forward pass, and writes
output/base20/load_smoke.json as the BASE-01 gate receipt (flat JSON, `status`
field, asserted fields — same convention as output/tinker/PROMOTED_*.json and
output/merge_v4_winner/merge_report.json).

Usage:
    python -m scripts.smoke_load_base20
    python scripts/smoke_load_base20.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same direct-execution sys.path fix as scripts/download_model.py — see that
# file's comment for why this is needed (scripts/ vs repo root on sys.path).
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_model import load_config  # noqa: E402

EXPECTED_MODEL_CLASS = "Qwen3_5MoeForConditionalGeneration"
CONFIG_V4_PATH = PROJECT_ROOT / "config" / "train_config_v4.yaml"
OUTPUT_PATH = PROJECT_ROOT / "output" / "base20" / "load_smoke.json"


def resolve_local_dir(config: dict) -> Path:
    """Resolve model.local_dir, relative to PROJECT_ROOT if not absolute."""
    local_dir = Path(config["model"]["local_dir"])
    return local_dir if local_dir.is_absolute() else PROJECT_ROOT / local_dir


def shard_stats(local_dir: Path) -> tuple[int, float]:
    """Return (shard_count, total_size_gb) for .safetensors files in local_dir."""
    shards = list(local_dir.glob("*.safetensors"))
    total_gb = sum(f.stat().st_size for f in shards) / (1024**3)
    return len(shards), total_gb


def write_receipt(status: str, **fields) -> dict:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"status": status, **fields}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(receipt, f, indent=2)
    return receipt


def run_smoke(config_path: Path = CONFIG_V4_PATH) -> dict:
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    import peft  # noqa: PLC0415
    import huggingface_hub  # noqa: PLC0415
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415

    config = load_config(config_path=config_path)
    local_dir = resolve_local_dir(config)
    shard_count, total_size_gb = shard_stats(local_dir)

    print(f"transformers {transformers.__version__}")
    print(f"peft {peft.__version__}")
    print(f"huggingface_hub {huggingface_hub.__version__}")

    versions = {
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
        "shard_count": shard_count,
        "total_size_gb": round(total_size_gb, 1),
    }

    print(f"Resolving config from {local_dir} (trust_remote_code=True) ...")
    model_config = AutoConfig.from_pretrained(str(local_dir), trust_remote_code=True)
    architectures = getattr(model_config, "architectures", None) or []
    model_class = architectures[0] if architectures else ""

    if EXPECTED_MODEL_CLASS not in model_class:
        return write_receipt(
            "fail",
            failing_field="model_class",
            model_class=model_class,
            host_notes=f"resolved architecture {model_class!r} does not contain {EXPECTED_MODEL_CLASS!r}",
            **versions,
        )

    print(f"Loading model from {local_dir} (bfloat16, device_map=cpu, trust_remote_code=True) ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir),
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Running a single forward pass on a 1-token input ...")
    input_ids = torch.tensor([[0]], dtype=torch.long)
    with torch.no_grad():
        model(input_ids)

    return write_receipt(
        "pass",
        model_class=model_class,
        host_notes="single forward pass on a 1-token CPU bf16 input executed without error",
        **versions,
    )


if __name__ == "__main__":
    try:
        result = run_smoke()
    except Exception as exc:  # noqa: BLE001 — smoke script: any exception is a gate failure
        write_receipt(
            "fail",
            failing_field="exception",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(f"BASE-01 SMOKE FAILED: {exc}")
        sys.exit(1)

    if result["status"] != "pass":
        print(f"BASE-01 SMOKE FAILED: {result}")
        sys.exit(1)

    print("BASE-01 SMOKE PASSED")
    print(json.dumps(result, indent=2))
