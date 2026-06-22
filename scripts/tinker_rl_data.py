#!/usr/bin/env python3
"""Tinker data adapter for Phase 9 RL rollout prompt pools.

Mirrors scripts/tinker_reasoning_data.py — same base model, same renderer,
same cookbook conventions — but loads prompts ONLY (no assistant target).

For RL the model generates completions at sampling time; the assistant turn
is empty (set by build_rl_prompts.py) and must NOT be pre-filled here.

EXPORTS (importable without a live Tinker session):
  BASE_MODEL    = "Qwen/Qwen3-30B-A3B"
  RENDERER_NAME = "qwen3_disable_thinking"
  load_rl_prompts(pool: str) -> list[dict]

Tinker import is LAZY (inside load_rl_prompts) so the module can be imported
in unit tests / CI environments where tinker is absent, matching the
tests/test_rl_train.py lazy-import convention.
"""

import json
import os

# ---------------------------------------------------------------------------
# Module-level constants (importable without tinker)
# ---------------------------------------------------------------------------
BASE_MODEL     = "Qwen/Qwen3-30B-A3B"
RENDERER_NAME  = "qwen3_disable_thinking"
MAX_LENGTH     = 8192

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GEN_TRAIN_PATH   = os.path.join(PROJ_ROOT, "data", "rl_prompts", "wp_gen_train.jsonl")
JUDGE_TRAIN_PATH = os.path.join(PROJ_ROOT, "data", "rl_prompts", "wp_judge_train.jsonl")

_POOL_PATHS = {
    "gen":   GEN_TRAIN_PATH,
    "judge": JUDGE_TRAIN_PATH,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rl_prompts(pool: str) -> list:
    """Return a list of prompt dicts for the requested pool.

    Each dict has the shape expected by the Tinker sampling client:
        {"messages": [{"role": "user", "content": "..."}]}

    The assistant turn is stripped — only the user prompt is returned so the
    model generates fresh completions at RL sampling time.

    Args:
        pool: "gen" or "judge"

    Returns:
        List of prompt dicts (user-turn only, no assistant target).

    Raises:
        ValueError: if pool name is unrecognised.
        FileNotFoundError: if the JSONL file does not exist (run
            scripts/build_rl_prompts.py first).
    """
    if pool not in _POOL_PATHS:
        raise ValueError(f"Unknown pool '{pool}'. Expected one of: {list(_POOL_PATHS)}")

    path = _POOL_PATHS[pool]

    # Primary path: manual JSONL load (no tinker dependency; works in tests +
    # CI; produces the exact dict format Tinker's sampling_client.sample() needs).
    prompts = _load_jsonl_prompts(path)

    # Tinker cookbook path (optional, for future integration with
    # FromConversationFileBuilder in prompt-only mode):
    # If the cookbook ever exposes a clean prompt-only path we can swap this
    # block in. For now, manual load is both simpler and dependency-free.
    # try:
    #     prompts = _load_via_cookbook(path)
    # except ImportError:
    #     prompts = _load_jsonl_prompts(path)

    return prompts


def _load_jsonl_prompts(path: str) -> list:
    """Load prompts from a JSONL file, stripping the empty assistant turn.

    Each row in the JSONL has:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": ""}]}

    We return only the user turn (prompt-only) to avoid pre-filling completions.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RL prompt file not found: {path}\n"
            "Run `python scripts/build_rl_prompts.py` to assemble the prompt pools."
        )

    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Extract user-only messages (drop empty assistant turn)
            user_msgs = [m for m in row["messages"] if m["role"] == "user"]
            if not user_msgs:
                continue
            prompts.append({"messages": user_msgs})

    return prompts


def _load_via_cookbook(path: str) -> list:
    """Optional: load via tinker-cookbook FromConversationFileBuilder.

    Requires tinker + tinker_cookbook to be installed. This path is provided
    for future integration only; the primary path is _load_jsonl_prompts.

    Tinker is imported LAZILY here so the module remains importable without it.
    """
    from tinker_cookbook import renderers  # noqa: PLC0415
    from tinker_cookbook.supervised.data import FromConversationFileBuilder  # noqa: PLC0415
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig  # noqa: PLC0415

    cc = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL,
        renderer_name=RENDERER_NAME,
        max_length=MAX_LENGTH,
        batch_size=1,
        # NONE = do not train on any assistant content (prompt-only mode)
        train_on_what=renderers.TrainOnWhat.NONE,
    )
    ds, _ = FromConversationFileBuilder(
        file_path=path, test_size=0, common_config=cc)()
    # Convert cookbook dataset items back to prompt dicts for the RL loop
    prompts = []
    for item in ds:
        if hasattr(item, "messages"):
            prompts.append({"messages": item.messages})
    return prompts


# ---------------------------------------------------------------------------
# CLI — quick smoke check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen_prompts   = load_rl_prompts("gen")
    judge_prompts = load_rl_prompts("judge")
    print(f"wp_gen pool   : {len(gen_prompts)} prompts loaded")
    print(f"wp_judge pool : {len(judge_prompts)} prompts loaded")
    if gen_prompts:
        print(f"  gen[0] user content (first 80 chars): {gen_prompts[0]['messages'][0]['content'][:80]!r}")
    if judge_prompts:
        print(f"  judge[0] user content (first 80 chars): {judge_prompts[0]['messages'][0]['content'][:80]!r}")
    print("TINKER-RL-DATA OK")
