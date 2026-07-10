"""_verify_reasoning_config.py — throwaway helper to assert train_config_reasoning.yaml values.

Exits non-zero with a message on any assertion failure. Used by Task 2 <automated> verify line.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
    import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "train_config_reasoning.yaml"


def fail(msg: str) -> None:
    print(f"FAIL  {msg}")
    sys.exit(1)


def main() -> None:
    if not CONFIG_PATH.exists():
        fail(f"config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)

    training = cfg.get("training", {})
    model = cfg.get("model", {})
    lora = cfg.get("lora", {})

    # RTRN-01: LR 2.0e-5
    lr = training.get("learning_rate")
    if lr != 2.0e-5:
        fail(f"training.learning_rate={lr!r}  expected 2.0e-5")

    # RTRN-02: max_seq_length 8192
    max_seq = model.get("max_seq_length")
    if max_seq != 8192:
        fail(f"model.max_seq_length={max_seq!r}  expected 8192")

    # D-01/D-04: local_dir basename must be qwen3-30b-wp-30_70-merged
    local_dir = model.get("local_dir", "")
    local_dir_basename = Path(local_dir).name
    if local_dir_basename != "qwen3-30b-wp-30_70-merged":
        fail(
            f"model.local_dir basename={local_dir_basename!r}  "
            f"expected 'qwen3-30b-wp-30_70-merged'"
        )

    # D-04: must NOT reference the stale merged-30_70 legacy dir
    if "merged-30_70" in local_dir:
        fail(
            f"model.local_dir={local_dir!r}  must NOT contain 'merged-30_70' "
            f"(stale legacy dir forbidden)"
        )

    # LoRA hyperparameters (adapter_config.json is the source of truth)
    r = lora.get("r")
    if r != 32:
        fail(f"lora.r={r!r}  expected 32")

    lora_alpha = lora.get("lora_alpha")
    if lora_alpha != 64:
        fail(f"lora.lora_alpha={lora_alpha!r}  expected 64")

    lora_dropout = lora.get("lora_dropout")
    if lora_dropout != 0.0:
        fail(f"lora.lora_dropout={lora_dropout!r}  expected 0.0")

    # target_parameters present
    target_params = lora.get("target_parameters")
    expected_tp = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    if target_params != expected_tp:
        fail(
            f"lora.target_parameters={target_params!r}  "
            f"expected {expected_tp!r}"
        )

    # modules_to_save must NOT be present (Pitfall 4)
    if "modules_to_save" in lora:
        fail(
            f"lora.modules_to_save={lora['modules_to_save']!r}  "
            f"must be ABSENT (embeddings already baked into merged base)"
        )

    # warmup_steps absolute (not warmup_ratio degenerating)
    warmup_steps = training.get("warmup_steps")
    if warmup_steps != 20:
        fail(f"training.warmup_steps={warmup_steps!r}  expected 20")

    # output_dir basename (FRESH dir, not the existing 30_70 adapter)
    output_dir = training.get("output_dir", "")
    output_basename = Path(output_dir).name
    if output_basename != "qwen3-30b-wp-30_70-reasoning":
        fail(
            f"training.output_dir basename={output_basename!r}  "
            f"expected 'qwen3-30b-wp-30_70-reasoning'"
        )

    print("_verify_reasoning_config: ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
