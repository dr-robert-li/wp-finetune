# Build brief (all workstreams, phase 3)

Repo: /home/rli/Desktop/projects/wp-finetune. Harness root: harness/.
Read first: harness/SPEC.md, harness/PLAN.md (Contracts, your workstream's Locked plan entry, and
"Binding changes 1-16", which OVERRIDE the locked-plan text), harness/INVENTORY.md, and the research
file(s) named in your task.

Rules:
- Edit ONLY the files you own (listed in your task). Read anything.
- Shared files are pre-created and READ-ONLY for you: harness/pyproject.toml, harness/wpjudge/__init__.py,
  harness/wpjudge/config.py (use `from wpjudge.config import load` / `Config`; every knob lives there).
  If you need a change there, say so in your report; do not edit.
- Self-contained: copy repo code in; never import from scripts/ or eval/.
- Ponytail style: the laziest correct solution, reuse prototypes, no speculative abstractions, no extra deps.
- Leave ONE runnable check (pytest file) for non-trivial logic. You may run it to make it work, but formal
  validation is done later by a separate evaluator agent, so do not claim it is validated.
- Python: harness/.venv/bin/python (package installed editable with test extras). Do NOT pip install into
  /home/rli/work/venv. Run tests as: cd harness && .venv/bin/python -m pytest tests/<file> -q
- Do NOT commit. Do not touch .planning/.
- GPU is shared: only the workstream told it may run a model server does so; kill it after; check nvidia-smi.
- Docker needs: sg docker -c 'docker ...'

Output: write your report to harness/build_notes/<WS>.md (files written, line counts, decisions, deviations
from plan with reasons, open questions, how to run the check). Your final reply must be just "done"
(final reply text has been arriving empty, so the file is the real channel).
