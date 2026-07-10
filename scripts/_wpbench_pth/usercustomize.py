"""Auto-imported at interpreter startup (when this dir is on PYTHONPATH).

Replaces wp-bench's ModelInterface.generate so REVL-04 invokes the Qwen3 model
the SAME way plan-02's accepted fidelity proof did: with
chat_template_kwargs={"enable_thinking": false}.

The Qwen3 chat template enables thinking by DEFAULT. With thinking on, v3 opens
<think> and, on generation-style prompts, never closes it — it writes the answer
INSIDE an unterminated think block (verified: eval_gen captured 15/17 responses
with <think> and 0/17 with </think>). wp-bench then either times out waiting on
the long generation, or scores a "<think>\n..." prefix that php_lint (execution
tasks) and `answer.upper().startswith(correct)` (knowledge tasks, core.py:134)
both reject. Disabling thinking yields clean, bounded, direct code — identical in
shape to the terse baseline. Applied to BOTH serves via the same monkeypatch =>
symmetric, fair comparison (and a harmless no-op on a model whose template
ignores the kwarg).

wp-bench's models.py calls litellm.completion() WITHOUT extra_body, so we cannot
post-process our way to enable_thinking=false — we replace generate() to thread
the kwarg through. The call mirrors models.py exactly (same sampling params) plus
extra_body. A defensive <think>...</think> pair-strip remains for any residual.
"""
import re

try:
    from litellm import completion
    from wp_bench import models as _m

    _THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

    def _generate(self, prompt: str) -> str:
        resp = completion(
            model=self.config.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=self.config.request_timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        out = resp.choices[0].message["content"] or ""
        return _THINK.sub("", out).strip()

    _m.ModelInterface.generate = _generate
except Exception:  # noqa: BLE001 - never break the interpreter on import
    pass
