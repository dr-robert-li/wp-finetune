# Evaluator brief (phase 4)
Repo /home/rli/Desktop/projects/wp-finetune, harness in harness/. You are an INDEPENDENT evaluator; you did
not build this. Be adversarial but fair: judge against harness/SPEC.md, harness/PLAN.md (Contracts, the
workstream's Locked plan entry, and Binding changes 1-16 which override it), and the builder's report
harness/build_notes/<WS>.md.
- Read the code fully. Run its check (harness/.venv/bin/python -m pytest tests/<file> -q) and add your own
  throwaway probes in the session scratchpad (not in harness/). Do not edit harness files.
- Check: contract conformance (exact dict keys/signatures other workstreams rely on), binding changes,
  edge cases, security guards, over-engineering (ponytail: flag needless code), and that the check is real.
- No model server / GPU use. Docker builds allowed only for W5 (sg docker -c '...'); do not start GPU containers.
Output: harness/build_notes/EVAL_<WS>.md with first line `VERDICT: PASS` or `VERDICT: FAIL`, then a numbered
defect list (severity blocker/major/minor, file:line, what, suggested fix) and what you ran. Minor-only => PASS.
Final reply: just "done".
