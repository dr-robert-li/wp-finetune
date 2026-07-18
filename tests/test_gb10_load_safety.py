"""Durable guard against the GB10 unified-memory `device_map="auto"` OOM trap.

Root cause (Phase 25-01 profiler, PID 2474645 OOM-killed at ~62% of a 67 GiB
load): on a GB10 the GPU and CPU are ONE physical 121 GiB pool, but
torch.cuda.mem_get_info() reports the whole pool as GPU-free, so
`device_map="auto"` balances a big model across a phantom CPU+GPU pair and the
two placements collide into a global OOM. Every full-model load must instead go
through scripts.sieve_arch.gb10_load_kwargs() (single-device + low_cpu_mem_usage).

This test fails if any script under scripts/ reintroduces a bare
`device_map="auto"` on a from_pretrained call, or if the helper's contract drifts.
AST-based, so comments and docstrings that merely mention the trap don't trip it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _auto_device_map_calls(tree: ast.AST) -> list[int]:
    """Return line numbers of any call passing device_map="auto" as a keyword."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "device_map" and isinstance(kw.value, ast.Constant) and kw.value.value == "auto":
                hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("py_file", sorted(SCRIPTS_DIR.glob("*.py")), ids=lambda p: p.name)
def test_no_bare_device_map_auto(py_file: Path) -> None:
    """No script may load a model with the GB10-unsafe device_map="auto"."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    hits = _auto_device_map_calls(tree)
    assert not hits, (
        f'{py_file.name} uses device_map="auto" at line(s) {hits} — GB10 OOM trap. '
        f"Use **sieve_arch.gb10_load_kwargs() instead."
    )


def test_gb10_load_kwargs_contract() -> None:
    """The helper must yield single-device placement + streaming, never "auto"."""
    from scripts import sieve_arch

    kw = sieve_arch.gb10_load_kwargs()
    assert set(kw) == {"device_map", "low_cpu_mem_usage"}, kw
    assert kw["low_cpu_mem_usage"] is True
    # Single destination: exactly one entry keyed "" (whole model, one device).
    assert list(kw["device_map"].keys()) == [""], kw
    assert kw["device_map"][""] in (0, "cpu"), kw
    assert kw["device_map"] != "auto"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
