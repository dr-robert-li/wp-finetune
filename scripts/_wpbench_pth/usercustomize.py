"""Auto-imported at interpreter startup (when this dir is on PYTHONPATH).

Monkeypatches wp-bench's ModelInterface.generate to strip Qwen3 <think>...</think>
scaffolding from model output before wp-bench scores it. wp-bench's knowledge
scorer does `answer.upper().startswith(correct_answer)` (core.py:134); the empty
`<think>\n\n</think>` scaffold the Qwen3 chat template emits would prefix every
answer and break the match for BOTH the baseline and reasoning serves alike.

This mirrors eval_gen.py:60 exactly so wp-bench consumes the model identically to
the project's own harness. Applied symmetrically => fair REVL-04 comparison.
"""
import re

try:
    from wp_bench import models as _m

    _THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
    _orig_generate = _m.ModelInterface.generate

    def _generate(self, prompt: str) -> str:
        return _THINK.sub("", _orig_generate(self, prompt)).strip()

    _m.ModelInterface.generate = _generate
except Exception:  # noqa: BLE001 - never break the interpreter on import
    pass
