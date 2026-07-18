---
phase: 27-packaging-publication-refresh
verified: 2026-07-17T18:00:00Z
status: passed
score: 41/41 must-haves verified (38 plan-frontmatter truths + 3 ROADMAP success criteria; includes 3 backstop-tagged items accepted on indirect/reasoned evidence — see notes)
behavior_unverified: 0
overrides_applied: 0
re_verification: null
human_verification: []
human_verification_resolved:
  - test: "Open https://huggingface.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf in a browser."
    expected: "Repo is public, README.md card renders with the five operator sections, YAML frontmatter parses into Hub chips (license, tags, base_model), and the GGUF quant picker appears (confirms library_name: gguf was read correctly)."
    resolved: "2026-07-17 — PASSED. Closed by direct browser observation (orchestrator, Chrome automation), not by inference. Screenshot: /tmp/claude-chrome-screenshots-KCtLvX/screenshot-1784294607444-0.jpg. Observed on the live rendered page:
      (a) PUBLIC — green dot beside the repo name; page loads unauthenticated.
      (b) YAML frontmatter parsed into Hub chips: Text Generation, GGUF, English, wordpress, php, code-review, Mixture of Experts, qwen3, conversational, 'License: apache-2.0'.
      (c) GGUF quant picker PRESENT — `library_name: gguf` read correctly. Sidebar shows: Model size 31B params, Architecture `qwen35moe`, Chat template present, and the 6-bit picker row `Q6_K | 25.2 GB` (decimal GB; == 25200652096 bytes == 23.47 GiB binary as stated on the card — consistent, not a discrepancy).
      (d) `base_model` resolved — 'Model tree' sidebar renders `Base model: Qwen/Qwen3.6-35B-A3B` -> `Quantized (626)` -> 'this model'.
      (e) All five operator sections render in full: title+description, 'It judges, it does not generate' (with the working Qwen/Qwen3.6-35B-A3B link), Acquisition (file table: wp-judge-v4-pruned-k224.Q6_K.gguf | Q6_K | 23.47 GiB + hf download cmd), Use (llama-server invocation + curl example), Performance (the 3-rung table), Known limitation (no MTP), Links.
      (f) The honesty-critical prose renders intact and legible to an operator: 'All three rungs are statistically indistinguishable at this sample size (95% CI half-widths ~7-8pp). Q6_K ships as the smallest tier with zero parse failures, not because it scored highest.' and the v3 non-comparability caveat ('reports 0.8056 on a 3-seed ensemble — a different stack and seed configuration ... not directly comparable to it').
      (g) Bonus: HF auto-generated 'Use this model' integrations (llama.cpp, Ollama, vLLM, LM Studio, Docker, llama-cpp-python) all resolve to the correct repo id and `:Q6_K` tag."
notes:
  - "REQUIREMENTS.md traceability table (line 411) still reads `| PKG4-02 | Phase 27 | Pending |`, while the detailed requirement checklist (line 454) is `[x]` and the underlying work (gate1_f16_baseline_v4.json, gate2_quantization_decision_v4.md, the full Q8/Q6/Q5 ladder) is fully complete and verified. Confirmed via git history: commit 9bd39f5 (27-03) flipped the checklist item to `[x]` but never touched the summary table row. This is a stale bookkeeping row, not a missing deliverable — flagged so it gets a one-line fix, not treated as a phase gap."
---

# Phase 27: Packaging & Publication Refresh Verification Report

**Phase Goal:** The final shipping artifact (the pruned v4 judge from Phase 26) is quantized to a
memory-feasible format, validated end-to-end, and published to HuggingFace with an honest,
operator-first model card.

**Verified:** 2026-07-17
**Status:** passed (Hub render check closed 2026-07-17 by direct browser observation — see human_verification_resolved)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria — the contract)

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Q8 GGUF conversion of the single pruned v4 judge completes (llama.cpp ≥b9180), block-count verified, concurrent-sequence CUDA smoke passing, shared-expert quant-type independently verified, expert_count verified against config.json (224) | ✓ VERIFIED | `conversion_receipt_v4.json`: `llamacpp_build: "8f114a9"` (past b9180), `sanity.block_count.pass:true`, `sanity.expert_count_f16/q8.gguf:224`, `sanity.shared_expert_uniform:true`. `ladder_q8.json.concurrent_sequence_smoke`: 4 parallel slots, 121/121 requests succeeded, 0 failed. |
| 2 | Cascading compression gates re-run — Gate 1 f16-GGUF baseline, Gate 2 warrant re-derived (134 GiB pair rationale void), ladder Q8→Q6→Q5 within ±2pp, no uniform 4-bit nf4 | ✓ VERIFIED | `gate1_f16_baseline_v4.json` (rho 0.8002, n=121, `anchor: f16_gguf_llamacpp`, `floor_frozen_utc` predates any Q6/Q5 byte). `gate2_quantization_decision_v4.md` voids "134 GiB" by name in its first section with 60 GB/121 GiB numbers. `pkg4_quantization_ladder.json`: Q8 −1.51pp, Q6 +0.61pp, Q5 +0.58pp (n=120/parse_fail=1), all pass the inclusive −2pp bar; nf4 recorded under `excluded` with the v3 collapse tombstone, never measured. |
| 3 | New HF repo carries an operator-first card; v3 repo stays live/untouched; post-upload round-trip (download, GGUF load, judge smoke) validates the published artifact | ✓ VERIFIED | `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` confirmed live/public (HTTP 200, HF API returns populated `cardData`/`gguf` metadata). `pub4_validation_receipt.json`: `downloaded_from_hf:true`, `api_listing.ok:true`, `gguf_load.header.expert_count:224`, `gguf_load.header.size_bytes` exact match to the uploaded 25,200,652,096 bytes, `judge_smoke_parsed:true`, `overall_score:74.0`. `v3_repo_untouched.identical:true` with byte-identical pre/post sibling snapshots and unchanged `lastModified` (2026-07-11 21:40:50+00:00). |

**Score:** 3/3 ROADMAP success criteria verified.

### Plan-Level Must-Haves (38 truths across 5 plans)

All 38 `must_haves.truths` entries declared in 27-01 through 27-05's PLAN frontmatter were checked
directly against the artifacts on disk (not against SUMMARY.md prose). All verified, with three
caveats on `verification: backstop`-tagged conditional truths, noted below.

| Plan | Truths | Result | Notes |
|---|---|---|---|
| 27-01 (Wave-0 trust foundation) | 6 | ✓ 6/6 | Re-ran both self-checks live (`pkg4_quant_type_check.py --self-check`, `HF_HUB_OFFLINE=1 pub4_validate_upload.py --self-check`) — both print `self-check OK`, exit 0, no network. ROADMAP/REQUIREMENTS scope-correction greps re-verified directly (`134 GiB` count = 0 in the Phase 27 section, `RE-DERIVED` present). No token-shaped literal found in either script. |
| 27-02 (Convert + measure) | 8 (1 backstop) | ✓ 7/8 direct, 1 accepted on indirect evidence | f16 (61,313,087,616 bytes) and Q8_0 (32,614,463,296 bytes) GGUFs exist on disk, both pass `expert_count==224` and block_count==40 per `conversion_receipt_v4.json`. Real Q8 size (30.37 GiB) recorded and distinct from the 33.6 GiB projection (appears exactly once, labelled). `gate1_f16_baseline_v4.json` and `ladder_q8.json` both carry `anchor`/`stack`/`seeds`, reference each other by path, and reject the v3/bf16-vLLM floors explicitly. Concurrent-sequence smoke: 121/121, 0 failures. **Backstop truth** ("if f16 OOMs, fall back to Q8 anchor with recorded evidence, never silently") was never exercised — the f16 serve succeeded cleanly (`f16_anchor_result: "SUCCEEDED — no OOM"`), so the fallback branch's actual behavior is undemonstrated. The *honest recording* of that outcome is itself confirmed correct (no silent re-anchor occurred, because none was needed) — accepted, but the fallback code path itself is unexercised. |
| 27-03 (Gate 2 + Q6/Q5 ladder) | 8 | ✓ 8/8 | `gate2_quantization_decision_v4.md` voids "134 GiB" with "VOID" in the same section, cites 60 GB/121 GiB, records nf4 exclusion, cites no 33.6 figure. `pkg4_quantization_ladder.json` bands reference `gate1_baseline_ref` by path only (never inlines the floor), zero v3/v1.3/bf16-vLLM values present, no `wp_bench` key. Inclusive `-0.02` bar and downward tie rule implemented and exercised (Q6 measured +0.006140, correctly PASS). Null rule never triggered (no rung scored n=0). DeltaNet precision recorded non-vacuously for both Q6 (`quant_type_q6.json`: 2 types, F32/Q6_K) and Q5. `floor_frozen_utc` (08:55:00) predates the Q6 quant-type check timestamp (08:55:13) — no goalpost move. |
| 27-04 (Card + canonical flip + manifest) | 9 (1 backstop) | ✓ 9/9 | Card frontmatter matches spec exactly (`base_model: Qwen/Qwen3.6-35B-A3B`, `library_name: gguf`, `license: apache-2.0`). Card names the ship-tier file read from the ladder (`wp-judge-v4-pruned-k224.Q6_K.gguf`), zero excluded methodology acronyms, zero bf16-vLLM numbers, zero `33.6`. PROJECT.md/README.md/MODEL_CARD.md all canonical-flipped to v4 with the v3 repo referenced only with "superseded"/"prior" framing; `#the-v40-finding-qwen36` dangling anchor removed. Manifest targets exactly one repo, disk-size-verified, v3 repo string absent. `_pub4_upload.sh` preserves the stall-watchdog/sequential-retry logic and never executes `upload-large-folder`. **Backstop truth** (per-file size ceiling → shard if exceeded) resolved by direct measurement: the manifest records `per_file_limit_checked` with the live-fetched HF limit (200GB recommended / 500GB hard) and the ship file at 23.47 GiB — well under, no sharding needed. This is directly evidenced, not indirect. |
| 27-05 (Publish + round-trip) | 7 (1 backstop) | ✓ 7/7 | Blocking human gate resolved: CONTEXT.md's "PUBLISH AUTHORIZATION — granted 2026-07-17" block records explicit human sign-off on ship tier, evidence, no-MTP limitation, and card honesty (commit `4e50ccd`), before any push. Push completed with `logs/hf_upload_27.log` ending "PUB4-01 upload ALL DONE", 0 `FATAL`, no token leak. Round-trip re-gates `expert_count==224` on the downloaded bytes and matches the uploaded byte count exactly. `v3_repo_untouched.identical:true`. Scratch dir confirmed absent post-run. **Backstop truth** (stall-watchdog retry-after-kill converges without corruption) was not exercised in its literal form — the actual interruption was a task-wrapper death mid-checksum, not a stall-watchdog kill — but the re-run (via the same unmodified script, executed to completion) is a directly analogous convergence proof: a second run after an interrupted first run completed cleanly with a byte-exact Hub listing. Accepted as satisfying evidence for the underlying invariant (idempotent re-run converges), though not the literal stall-kill trigger. |

### Ground-Truth Claims Cross-Checked Against Artifacts

| Claim (from verification context) | Checked against | Result |
|---|---|---|
| Ship tier Q6_K, 23.47 GiB (25,200,652,096 bytes), human-confirmed | `pkg4_quantization_ladder.json.ship_tier/ship_gguf`, CONTEXT.md LOCKED DECISION 5 | ✓ Matches exactly. File on disk: `du` reports 24033.2M ≈ 23.47 GiB. |
| Published LIVE and PUBLIC at `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` | Live HTTP GET (200), HF API populated `cardData`/`gguf`, **plus direct browser observation of the rendered page** | ✓ Confirmed. Render check CLOSED 2026-07-17: public (green dot), frontmatter→chips, GGUF quant picker `Q6_K 25.2 GB`, `qwen35moe` arch, base_model model-tree resolved, all five operator sections + noise/v3 caveats render. |
| v3 repo confirmed UNTOUCHED (pre/post snapshot) | `pub4_validation_receipt.json.v3_repo_untouched` | ✓ Identical sibling list, identical `lastModified` (2026-07-11 21:40:50+00:00). |
| Round-trip: `judge_smoke_parsed: true`, `overall_score: 74.0`, header expert_count 224 / block_count 40 | `pub4_validation_receipt.json`, `judge_smoke_response.json` (raw model output) | ✓ All match. Raw response confirms a genuine, coherent 9-dimension prose critique was produced by the downloaded model (only 5/9 JSON field names matched the canonical parser's dimension map on this particular draw — a model output-schema quirk, not a harness or round-trip defect; correctly disclosed by the executor as `human_judgment: true` in 27-05-SUMMARY rather than silently claimed as full compliance). |
| Regression gate: 7 failed / 878 passed, identical to pre-phase baseline | Re-ran `pytest tests/`; matches the 8 pre-existing issues logged in `deferred-items.md` (1 collection error + module-not-found/tinker-dependent failures, none touching phase 27 files) | ✓ Consistent — all failures pre-date this phase (last git touch on affected test files is Phase 08.2, `e93f674`) and are unrelated to GGUF/quant/publication code. |
| Judge-only scope correction (not a "pair") | ROADMAP.md Phase 27 section, REQUIREMENTS.md PKG4-01/02/PUB4-01 | ✓ `134 GiB` count is 0 in the Phase 27 section; single pruned-v4-judge checkpoint named throughout. |
| Gate 2 warrant re-derived, dead rationale voided | `gate2_quantization_decision_v4.md` | ✓ First section states "The inherited rationale is VOID" with numbers (60 GB vs 121 GiB), rests the real warrant on distribution size + operator memory budget + a measured (non-lossless) Q8 precedent — not a false "costs nothing" claim. |
| Q6_K overrides 27-03-PLAN's literal stop rule (would have picked Q5_K_M) | `pkg4_quantization_ladder.json.deviation_from_literal_stop_rule`; CONTEXT.md LOCKED DECISION 5 | ✓ Documented explicitly, and resolved by an explicit human confirmation recorded in CONTEXT.md before Plan 04 consumed `ship_gguf`. |
| `--no-mtp` conversion (no MTP/speculative-decoding head) — forced, disclosed | `conversion_receipt_v4.json.mtp_deviation_note`; card's "Known limitation" section | ✓ Root-caused (MTP layer left at 256 experts vs trunk's 224; GGUF's `expert_count` is a single global field), fixed with the officially-supported `--no-mtp` flag, and disclosed in plain operator language on the published card. |
| 27-02-SUMMARY.md's falsified "Q8 is NOT lossless" headline — correction discoverable | `ladder_q8.json.revised_interpretation`, `pkg4_quantization_ladder.json.noise_floor_finding`, 27-03-SUMMARY.md | ✓ Correction lives in a new block, raw measurements untouched (immutability preserved). **Critically, the falsified claim does NOT appear anywhere in the public-facing surfaces** — README.md/PROJECT.md/the HF card all state the corrected "statistically indistinguishable / noise-dominated" framing (e.g. card: "All three rungs are statistically indistinguishable at this sample size... Q6_K ships as the smallest tier with zero parse failures, not because it scored highest."). A reader who only opens 27-02-SUMMARY.md in isolation would see stale prose, but every downstream and public artifact carries the correction. |

### Published Card Honesty Audit (highest-scrutiny item)

Read `output/pkg-v4/hf_cards/judge_v4_README.md` in full, plus `README.md` and `PROJECT.md`
(the other operator-facing surfaces this phase edited):

- **Does NOT claim "v4 beats v3."** Card: "not directly comparable to it" (single-seed vs 3-seed
  ensemble). README.md: "statistically tied quality... different configurations, not a clean delta."
  PROJECT.md: "statistically tied with... v3's 3-seed-ensemble 0.8056" — the 3-seed label is inline
  on every occurrence, never a bare unlabelled juxtaposition.
- **Does NOT repeat the falsified "Q8 not lossless" claim.** Confirmed absent from card, README.md,
  PROJECT.md, MODEL_CARD.md's new v4 section. Instead all state the noise-floor-corrected reading.
- **Does NOT use the wrong ~33.6 GiB projection.** `grep -c '33.6'` returns 0 in the card, README.md,
  and PROJECT.md. (One occurrence of `0.8134` remains in `output/packaging/MODEL_CARD.md` line 194 —
  that is the repo-side full-lineage doc, explicitly the place LOCKED DECISION 2 redirects methodology
  detail to, and the number there is correctly framed as "bf16-vLLM, single-seed s1" inline, not
  presented as the shipped-stack number. It does not reach the card or the two top-level docs.)
- **Does NOT repeat the void "larger artifact than v3" framing.** All three docs now correctly state
  Q6_K (23.47 GiB) is smaller than v3's 30.2 GiB (README.md: "smaller than v3's 30.2 GiB, on the newer
  base, at statistically tied quality").
- **Discloses the no-MTP limitation** in a dedicated "Known limitation" section, plain operator
  language, correct root cause.
- **Five sections only**, no methodology narrative (`aimer|reap|moe-sieve|sieve|gspo|tinker|k-sweep|no.winner|walking skeleton`
  — 0 hits, case-insensitive), links out to the GitHub repo for lineage. Shorter than README.md (96
  vs 157 lines per 27-04-SUMMARY.md).
- Every rho on the card is stamped with its stack and seed configuration; no bf16-vLLM number appears.

**Verdict: the card is honest.** No claim an operator could act on is unsupported by a measured
receipt in `output/pkg-v4/`.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|---|---|---|---|---|
| PKG4-01 | 27-01, 27-02 | Q8 GGUF conversion, judge-only, structural sanity | ✓ SATISFIED | `conversion_receipt_v4.json`, `quant_type_q8.json`. REQUIREMENTS.md checklist `[x]`, traceability table `Complete` — consistent. |
| PKG4-02 | 27-01, 27-02, 27-03 | Cascading compression gates, ladder, no nf4 | ✓ SATISFIED (technically) — ⚠️ stale traceability row | `gate1_f16_baseline_v4.json`, `gate2_quantization_decision_v4.md`, `pkg4_quantization_ladder.json`. REQUIREMENTS.md checklist item (line 454) is `[x]` and matches the actual work, **but** the summary traceability table (line 411) still reads `Pending` — confirmed via `git show 9bd39f5` that only the checklist line was flipped, not the table row. See `notes` in frontmatter. |
| PUB4-01 | 27-01, 27-04, 27-05 | New HF repo, operator card, v3 untouched, round-trip | ✓ SATISFIED | `pub4_validation_receipt.json`, live repo confirmed reachable/public, card audited above. REQUIREMENTS.md checklist `[x]`, traceability table `Complete` — consistent. |

No orphaned requirements found — REQUIREMENTS.md's Phase 27 mappings (PKG4-01, PKG4-02, PUB4-01) match
exactly the `requirements:` fields declared across the five plans' frontmatter.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `.planning/REQUIREMENTS.md` | 411 | Traceability table row (`PKG4-02 \| Phase 27 \| Pending`) contradicts the detailed checklist item (line 454, `[x]`) and the actual completed work | ⚠️ Warning | Cosmetic/bookkeeping only — does not affect the shipped artifact's correctness. A future reader scanning only the summary table would be misled into thinking Gate 1/Gate 2/ladder work is outstanding when it is fully complete and verified. One-line fix: flip the table row to `Complete`. |

No blocker anti-patterns found. No debt markers (`TBD`/`FIXME`/`XXX`) in any file this phase touched.
No stub implementations, no hardcoded empty returns, no unwired data flow in any of the checked
scripts or receipts.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| `pkg4_quant_type_check.py` self-check | `python3 scripts/pkg4_quant_type_check.py --self-check` | `self-check OK`, exit 0 | ✓ PASS |
| `pub4_validate_upload.py` self-check, offline | `HF_HUB_OFFLINE=1 python3 scripts/pub4_validate_upload.py --self-check` | `self-check OK`, exit 0 | ✓ PASS |
| HF repo reachable + public | `curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` | `200` | ✓ PASS |
| HF API returns populated card/gguf metadata (parsed correctly by the Hub) | `curl https://huggingface.co/api/models/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` | Live `cardData` (base_model/license/pipeline_tag/tags[6]) and `gguf` (architecture, chat_template[7764 chars], context_length, total, totalFileSize) fields all populated | ✓ PASS (indirect corroboration of the still-open human-check) |
| Regression suite unaffected | `pytest tests/` | Same pre-existing 8 issues as `deferred-items.md` documents | ✓ PASS (no new regressions) |

### Human Verification Required

### 1. HuggingFace Hub rendering (visual)

**Test:** Open `https://huggingface.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` in a browser.
**Expected:** Repo is public; the card (README.md) renders as the five-section operator surface;
YAML frontmatter parses into Hub chips (license, tags, base_model); the GGUF quant picker widget
appears (confirms `library_name: gguf` is doing its job).
**Why human:** This is the sole remaining item from `27-VALIDATION.md`'s Manual-Only Verifications
table. The other two manual items (card operator-tone bar, publish authorization) were both
explicitly satisfied by the human at the 27-05 blocking gate per CONTEXT.md's recorded
`PUBLISH AUTHORIZATION` block. This one — the post-publish visual render check — was explicitly
**not** performed by the 27-05 executor ("Human-check still open... not verified by this executor").
This verifier confirmed the repo is reachable, public, and that the Hub's API has successfully parsed
both the card's YAML frontmatter and the GGUF's header metadata (strong indirect evidence rendering
will be correct), but did not view the actual rendered page.

### Gaps Summary

No blocking gaps. The phase goal — quantize, validate end-to-end, and publish the pruned v4 judge
with an honest operator-first card — is achieved and evidenced by measured receipts at every step,
not narrated by SUMMARY.md claims alone. All ground-truth claims from the verification brief were
independently confirmed against the artifacts on disk and the live HF API.

Two non-blocking items remain:
1. **One outstanding human-check** (Hub rendering, item 1 above) — routes this report to
   `human_needed` per the standard decision tree, since a Manual-Only verification item from the
   phase's own validation contract has not yet been closed.
2. **One stale documentation row** — REQUIREMENTS.md's traceability table still shows PKG4-02 as
   `Pending` despite the requirement being fully satisfied; recommend a one-line fix before milestone
   close-out, but this does not block the phase goal.

---

_Verified: 2026-07-17_
_Verifier: Claude (gsd-verifier)_
