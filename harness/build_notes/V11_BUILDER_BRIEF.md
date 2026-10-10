# v1.1 builder brief (shared by WS-A..D)
You are the builder for one workstream (named in your prompt). You run inside your own git worktree.
1. First: `git merge --ff-only main` so the worktree has commit a6d598f (plan + research). Read `harness/PLAN_v1.1.md` fully: the "Rules for every agent", "Cross-task decisions", "Workstreams and file ownership", "Worktree setup", and your WS section. Read the research file(s) your WS cites (`harness/research/v11_r*.md`).
2. Worktree setup: `cd harness && python3 -m venv .venv && .venv/bin/pip install -e '.[test]'`. Check that `wpjudge.__file__` is in your worktree and that `.venv/bin/python -m pytest --version` works.
3. Edit only the files and functions your WS owns. Ponytail: smallest correct diff; match the surrounding style.
4. Unit tests: `.venv/bin/python -m pytest tests/` must pass. You may run your live e2e section to debug. Every GPU server runs under `flock /tmp/wpjudge-gpu.lock`, with `GGUF=/home/rli/Desktop/projects/wp-finetune/models/_gguf/shrink-iq4xs-im_f16.gguf`; kill it after use. B and C: also export `HOST_PARALLEL=1 WPJUDGE_WORKERS=1`.
5. Commit on your worktree branch with `git -c user.name="Robert Li" -c user.email=dr.robert.li.au@gmail.com commit`. Add no Co-Authored-By or "Generated with" lines. Do not touch main.
6. Write `harness/build_notes/V11_WS<X>.md`: what changed (file:function), test results, the live run evidence, deviations from the plan, and the branch name plus the worktree path. Reply only "done".
