# Docs review: shrink study ship (2026-10-10)

**Verdict: PASS, after the fixes below. Nothing is committed or uploaded.**

All other numbers match the ground truth: sizes, MiB, tok/s, rho, CIs, verdict % and the derived percentages. The card's file names match the manifest `repo_path`. The card lists the 4 files that will be on the repo. All `llama-server` flags appear in the build 9954 `--help`. The "Expected output" sample matches the capture in the scratchpad (`req.json`, `resp.json`, `smoke_serve.log`). The small-file conditions are present. No stale claims remain.

## Fixes

1. Card (`output/pkg-v4/hf_cards/judge_v4_README.md`): the 160-expert cut has 19 unreadable outputs, not 7 (7 parse failures plus 12 not valid JSON).
2. `CHANGELOG.md`: same correction, 7 to 19 unreadable for the 160-expert cut.
3. `JOURNAL.md`: same correction, reworded to "19 unreadable answers (7 parse failures and 12 not valid JSON)".
4. Card stability table: the small-file q8_0 KV row now reads "1 (format break). Fails the bar. Do not use."
5. Card Limits: added that GPU memory was measured at a 5120-token context, and that 16384 uses more.
6. Card low-VRAM section: "quality stays the same" is now "The verdicts stayed the same (100% agreement)".
7. `README.md`: the small-file GPU figure "~12.3 GB" is now the measured "12314 MiB".
8. `README.md`: the quickstart `llama-server` command now turns thinking off with `--chat-template-kwargs '{"enable_thinking": false}'`.

## Open doubts

- `llama-cli` is not installed, so the card's `llama-cli` command is unrun. I checked `-st` and `--chat-template-kwargs` in `common/arg.cpp`.
- The Ollama `--think=false` flag, the `hf.co/...:IQ4_XS` pull and Ollama support for this model are unverified. The card does not claim it tested them.
- The sample output ran at a 5120-token context, the same as the measurements, not at the 16384 the card recommends.
- The old README curl example (outside the diff) uses a different prompt and `max_tokens` 8192, and does not turn thinking off. I left it.
- The final and stability reviews give slightly different CIs for IQ4_XS with the q8_0 KV cache ([0.909, 0.961] and [0.908, 0.960]). The docs use the stability numbers. The difference comes from bootstrap resampling.
