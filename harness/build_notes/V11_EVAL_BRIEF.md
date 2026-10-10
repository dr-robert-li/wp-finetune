# v1.1 workstream evaluator brief
You are an independent evaluator for one workstream. You did not build it. Do not fix code. Edit nothing except your report.
- Work in the worktree path given in your prompt (`cd <wt>/harness`). Use that worktree's `.venv`. Check that `wpjudge.__file__` is inside the worktree. If the worktree has no `.venv`, create it with `python3 -m venv .venv && .venv/bin/pip install -e '.[test]'`.
- Spec: your WS section of `PLAN_v1.1.md`, plus the cited research. Builder report: `build_notes/V11_WS<X>.md`. Diff: `git diff 5e5ea1a..HEAD`.
- Check:
  1. Every spec item is implemented as planned, and deviations are justified.
  2. The code is correct, smallest-diff, and matches the surrounding style. Look for edge-case bugs.
  3. The unit tests are meaningful, not performative. Run `.venv/bin/python -m pytest tests/`.
  4. **Run your WS's live e2e section yourself.** Run it under `flock /tmp/wpjudge-gpu.lock`, with `GGUF=/home/rli/Desktop/projects/wp-finetune/models/_gguf/shrink-iq4xs-im_f16.gguf` (B and C: also `HOST_PARALLEL=1 WPJUDGE_WORKERS=1`). Read the actual output and reports, not just PASS lines. Kill any server you started.
  5. Functional sanity: does the feature actually do what the user needs?
- Write `<wt>/harness/build_notes/EVAL_V11_WS<X>.md`: numbered findings (BLOCKER/MAJOR/MINOR, with evidence and a fix), the commands run and their results, and finally `VERDICT: PASS` or `VERDICT: FAIL` (FAIL if there is any BLOCKER or MAJOR). Reply only "done".
