#!/usr/bin/env python
"""Tinker pivot P1 — data adapter for the WP reasoning SFT.

Our reasoning dataset is already OpenAI chat format (JSONL with a `messages` key),
so the cookbook's built-in `FromConversationFileBuilder` consumes it directly — no
custom dataset class needed.

Decisions (see .planning/TINKER-PIVOT-RESEARCH.md):
- base/tokenizer : Qwen/Qwen3-30B-A3B (matches Phase 4.3 base architecture)
- renderer       : qwen3_disable_thinking — our format is IN-BAND prose + [/REASONING]
                   + <judge_output> JSON, NOT native <think> blocks, so we do not want
                   the thinking renderer injecting <think> scaffolding.
- train_on_what  : LAST_ASSISTANT_MESSAGE — every row is single-turn (user -> assistant);
                   ALL_ASSISTANT_MESSAGES warns (renderer lacks the extension property).
- special tokens : <wp_gen>/<wp_judge> were tokenizer special tokens in Phase 4.3
                   (train_config_reasoning.yaml). Tinker uses the STOCK Qwen3 tokenizer,
                   so they train as plain-text literals. Verified they survive the
                   tokenize/decode round-trip; the format-stability markers ([/REASONING],
                   <judge_output>) were already plain text and are unaffected.
"""
from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

BASE_MODEL = "Qwen/Qwen3-30B-A3B"
RENDERER_NAME = "qwen3_disable_thinking"
MAX_LENGTH = 8192
TRAIN_PATH = "data/reasoning_dataset/openai_train.jsonl"
VAL_PATH = "data/reasoning_dataset/openai_val.jsonl"


def build_datasets(train_path: str = TRAIN_PATH, val_path: str = VAL_PATH,
                   batch_size: int = 8):
    """Return (train_dataset, val_dataset) as cookbook SupervisedDatasets."""
    cc = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL,
        renderer_name=RENDERER_NAME,
        max_length=MAX_LENGTH,
        batch_size=batch_size,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    train_ds, _ = FromConversationFileBuilder(
        file_path=train_path, test_size=0, common_config=cc)()
    val_ds, _ = FromConversationFileBuilder(
        file_path=val_path, test_size=0, common_config=cc)()
    return train_ds, val_ds


if __name__ == "__main__":
    tr, va = build_datasets(batch_size=8)
    print(f"train batches: {len(tr)} | val batches: {len(va)} (batch_size=8)")
    b = tr.get_batch(0)
    print(f"batch[0]: {len(b)} Datums; first datum tokens={b[0].model_input.length}")
    print("DATA-ADAPTER OK")
