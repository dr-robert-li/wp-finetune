# Continue Here — Phase 04.3-03 (merge-vs-training discriminator)

**Paused:** 2026-06-04 · **Reason:** context exhaustion (clean stop) · **Branch:** main

## Current Position

Phase 04.3 EXECUTING. Plan **04.3-03** (the merge-vs-training discriminator) in progress.
**Task 1 of 4 code is DONE + committed WIP** (`1acd70d`); the **Task 1 `--binding-dryrun`
gate has NOT passed** (blocked on a GB10 unified-memory load envelope). Tasks 2 (3 capture
arms) + 3 (Wilson-CI verdict) not started.

Prior plans in this phase are CLOSED: 04.3-01 (train, SUMMARY), 04.3-02 (bisect, verdict
REQUIRES_ADDITIONAL_ITERATION). `04.3-REOPEN-PLAN.md` is a 0-task BRIEF — do NOT execute it.

## Completed this session

- 04.3-02 executed end-to-end → verdict REQUIRES_ADDITIONAL_ITERATION (ckpt-50 65% terse WORSE
  than ckpt-72 37%, disjoint Wilson CIs; targets 0% terse; collapse = base-prior revert).
  Commits 8af4513 / 25bc633 / 5b1c8e8 / e94e0f7.
- 04.3-03 planned + plan-checked (3 rounds, all BLOCKERs resolved) → `edecfea`.
- 04.3-03 Task 1 code: `checkpoint_parse_check.py` runtime MoE-binding guard
  (`probe_moe_binding`) + `--no-adapter`/`--include-streams`/`--max-new-tokens 2048`/`--out`
  histogram + user-only prompts; 9/9 GPU-free tests pass. Committed WIP `1acd70d`.

## Remaining work (04.3-03)

1. **Pass the Task 1 `--binding-dryrun` gate** (the live 30B probe) — see blocker below.
2. **CP1** human-verify GPU checkpoint (gate the 3 capture arms).
3. **Task 2**: 3 capture arms on identical cot+ctf @120 slice — merged-vLLM-bf16 (`--limit 120`),
   merged-Unsloth-4bit (`--no-adapter`), unmerged-Unsloth-4bit (`--checkpoint-dir`).
4. **Task 3**: Wilson-CI compare + engine cross-check → ONE `DISCRIMINATOR:` verdict in
   `04.3-03-RESULTS.md` + `04.3-03-SUMMARY.md`.

## NEXT DECISIVE ACTION

### UPDATE 2026-06-07 — PIVOT TO THINKING MACHINES **TINKER** (cloud LoRA); local-venue work below is OBVIATED

The GB10 cannot host Qwen3-30B-A3B bf16 in-process (load+adapter transient ~122 GiB > 124.6 GiB
total — proven across every loader/precision; even 112 GiB allowance + multi-user.target are
insufficient). **USER decision: pivot to Tinker** (Thinking Machines' managed cloud LoRA fine-tune
+ OpenAI-compatible sampling). This dissolves the entire local-memory problem class AND obviates
04.3-03 (merge-vs-training discriminator), the multi-user.target job, the 4-bit build, and the
04.4 merge — we re-train the reasoning LoRA in the cloud and sample/eval it directly (no merge).

Research + plan: `.planning/TINKER-PIVOT-RESEARCH.md`. **P0 DONE (2026-06-07):**
- `.venv-tinker` (gitignored) with `tinker` 0.22.3; `TINKER_API_KEY` in `.env` (gitignored, untracked).
- Auth OK; **`Qwen/Qwen3-30B-A3B` (+Base +Instruct-2507) accessible** (41 models); loop smoke PASS
  (`forward_backward`+`optim_step` on Llama-3.2-1B). Re-runnable: `scripts/_tinker_smoke.py --loop`.

**NEXT = P1 (data adapter + open decisions):** write a `ChatDatasetBuilder` over
`data/reasoning_dataset/openai_{train,val}.jsonl` (`conversation_to_datum`); RESOLVE the
special-token question (did Phase 4.3 add `<wp_gen>`/`<wp_judge>` to the tokenizer? Tinker uses
stock Qwen3); pick renderer (`qwen3` thinking vs `qwen3_disable_thinking` — the format-stability
lever REVL-05 failed on); choose base variant. Then P2 SFT (LoRA Qwen3-30B-A3B) → P3 eval terse
rate → P4 decide. Local `merged-v2`/`ckpt-72` are reference/fallback only (READ-ONLY, not promoted).

---
### UPDATE 2026-06-06 — bf16 DESKTOP-DIRECT LOAD ALSO BLOCKED (intrinsic); local PATH was multi-user.target detached job (SUPERSEDED by the Tinker pivot)

Executed 04.3-03 Task 1. The convergence-validated "bf16 is flat ~57 GiB, safe on the live
desktop" premise is **EMPIRICALLY FALSIFIED**. Three in-process loaders (Unsloth-bf16,
transformers-streaming-bf16 `device_map`+`low_cpu_mem_usage`, and a 12-shard re-shard) ALL peak
~100–103 GiB on the GB10 unified pool with GNOME up and trip the watchdog. python RSS ~34 GiB
fixed + ~65 GiB on the unified device → the peak is intrinsic to materializing 30B on the unified
pool (precision/loader/shard-size independent). Full evidence + trip table:
`output/format_stability/discriminator/MEMORY-INVESTIGATION-bf16.md`. Host recovered every time,
no reboot. **No `DISCRIMINATOR` verdict written — the binding guard never ran (load never
completed); this is an INFRA blocker, explicitly NOT `BINDING_FAILED`.**

Also fixed an env regression: the `unsloth-headless` container had **torchao 0.17.0** (auto-installed
Jun-4 23:46, needs torch≥2.11; we have 2.10.0a0nv) breaking the transformers import. Removed it
(`docker exec unsloth-headless pip uninstall -y torchao`; unused — bf16/bnb only). NOTE: a container
RESTART may reintroduce it — re-run that uninstall if the import breaks again.

**THE PATH (USER decision taken: build the detached multi-user.target job — DONE, commit 6d4bb18):**
The two PRIMARY arms need an in-process Unsloth `load_adapter`, which only completes under
`multi-user.target` (drop GNOME → ~6 GiB freed → the ~103–108 GiB peak fits; the Jun-5 pre-quant
completed there). The full run is scripted as ONE unattended detached root job.

USER LAUNCHES (needs sudo; kills this x11 session for ~1.5–2 hr, then desktop returns):
```
sudo systemd-run --unit=disc043 --collect --property=IgnoreOnIsolate=yes \
  bash /home/robert_li/Desktop/projects/wp-finetune/scripts/_discriminator_multiuser.sh
```
Watch from a TEXT TTY (Ctrl+Alt+F3, survives the isolate):
`journalctl -u disc043 -f`  and  `tail -f logs/discriminator_multiuser.log`

The job: Task1 binding-dryrun gate (a watchdog trip here ⇒ writes `INFRA_BLOCKER.txt`
`LOAD_INFEASIBLE_EVEN_ISOLATED` and cleanly restores the desktop — meaning even isolation is
insufficient → next step would be a true all-4bit ckpt or smaller hardware) → if BOUND, Task2
three bf16 arms SEQUENTIAL → Task3 → `DISCRIMINATOR:` verdict in `04.3-03-RESULTS.md` +
`output/format_stability/discriminator/summary.md`. trap restores graphical.target on any exit.

AFTER it finishes, RE-RESUME: read `04.3-03-RESULTS.md` for the verdict, then write `04.3-03-SUMMARY.md`
+ update STATE/ROADMAP (plan complete). The verdict SELECTS the corrective branch for a NEXT plan
(MERGE_ARTIFACT → rewrite the merge math; TRAINING_UNDERIMPRINT → corrective retrain; INCONCLUSIVE →
widen n; BINDING_FAILED → a load path that binds the MoE LoRA) — this plan does NOT execute it.

---
### UPDATE 2026-06-05 (convergence) — PLAN PIVOTED TO bf16 (SUPERSEDED for the VENUE only; science stands)

Plan `04.3-03-PLAN.md` was REPLANNED to a **bf16-only** measurement path (commit `bf5ab0a`)
after a cross-AI convergence loop (`04.3-REVIEWS.md`, cycles `fa5b231`→`9610c1b`): 3 HIGH
concerns (all "4-bit is invalid + unexecutable on this GB10 host"), converged to 0 HIGH after
the bf16 rewrite. Phase marked PLANNED.

**THE PATH (bf16, orchestrator-runnable — NO sudo, NO isolate, NO pre-quant):**
1. **Task 1 gate** — bf16 binding-dryrun in the `unsloth-headless` container (watchdog backstop):
   `--binding-dryrun --no-4bit --base models/qwen3-30b-wp-30_70-merged-v2 --checkpoint-dir checkpoint-72`
   (~57 GiB flat peak, no double-hold → SAFE on live desktop). Confirms load_adapter binds the
   fused-MoE experts (activation-delta guard → BOUND) or HALTS BINDING_FAILED.
2. **CP1** human-verify, then re-pre-register Wilson-CI thresholds at bf16 BEFORE capture.
3. **Task 2** — 3 arms, all bf16, loaded SEQUENTIALLY (never co-resident; two loads ≈114 GiB):
   merged-72-vLLM-bf16 (`--limit 120`), merged-72-Unsloth-bf16, unmerged-72-Unsloth-bf16
   (`from_pretrained(merged-v2)` + `load_adapter(checkpoint-72)`). Engine cross-check =
   vLLM-bf16 vs Unsloth-bf16 (pure engine; quantization-not-material prereq auto-satisfied).
   PRE-EXEC HARDENING (cycle-2 MEDIUM): before the ~57 GiB Unsloth load, confirm vLLM's ~63 GiB
   is released (wait-until-free-RAM ≥110 GiB interlock) — the two never co-reside.
4. **Task 3** — Wilson-CI compare → ONE `DISCRIMINATOR:` verdict.

**DISCARDED:** the 4-bit pre-quant artifact (`models/qwen3-30b-wp-30_70-merged-v2-4bit`,
attn-only nf4 + transformers-expanded experts incompatible with the Unsloth fused-MoE adapter;
no 4-bit ckpt for the merged arm) and the on-the-fly 4-bit load (~108 GiB double-hold + Qwen3-MoE
router-quant corruption). Do NOT chase the sudo/systemd-run pre-quant below.

Resume: `/gsd:execute-phase 04.3` (only 04.3-03 incomplete).

---
<details: SUPERSEDED 4-bit trail — kept for audit, do NOT execute>

### UPDATE 2026-06-05 ~00:10 — on-the-fly load RULED OUT (peak math); pre-quant bug FIXED

Re-resumed. Verified against PRIMARY-SOURCE watchdog logs (not the memory summaries):
`logs/binding_dryrun_watchdog.log` + `logs/tf4bit_watchdog.log` BOTH show `tripped=1` —
the ~20.3 GiB figure is the **watchdog TRIP line, NOT a natural trough**. Neither run
completed; `peak_used≈98.4 GiB` was measured *at the kill while still climbing* → true
completion peak ~108 GiB. Room above the 16 GiB cascade = 114.7−16 = **98.7 GiB < ~108 GiB**.
**CONCLUSION: with GNOME up there is NO watchdog floor that both completes an on-the-fly 4-bit
load AND protects the host.** The "lower the floor → it finishes" idea is falsified — do NOT
launch `--binding-dryrun` against the bf16 base on the live desktop under any floor.

The earlier pre-quant FAILURE was a FIXABLE systemd bug, not the memory wall:
`logs/prequant_mergedv2.log` (13:38:51) shows isolate→restore→Terminated in one second —
`systemctl isolate multi-user.target` stopped the transient `prequant-mergedv2` unit itself.
**FIX APPLIED** to `scripts/_prequant_multiuser.sh` header: launch with
`--property=IgnoreOnIsolate=yes` so the unit survives the isolate (+ documented a root-TTY
manual-isolate fallback).

**THE PATH (4-bit ckpt is on the critical path — Task 2's two Unsloth-4bit arms need it too):**
1. USER runs the FIXED detached pre-quant (needs sudo password — orchestrator cannot):
   ```
   sudo systemd-run --unit=prequant-mergedv2 --collect --property=IgnoreOnIsolate=yes \
     bash /home/robert_li/Desktop/projects/wp-finetune/scripts/_prequant_multiuser.sh
   ```
   Verify survival: `systemctl is-active prequant-mergedv2` stays active through the isolate;
   log reaches a `post-isolate avail=...` line. ~6-8 min; desktop drops then returns.
2. Orchestrator verifies `models/qwen3-30b-wp-30_70-merged-v2-4bit` loads cheaply in Unsloth
   (~16 GB) + `load_adapter(checkpoint-72)` works.
3. `--binding-dryrun --base models/qwen3-30b-wp-30_70-merged-v2-4bit` → ~16 GB, SAFE on live
   desktop → Task 1 gate passes.

OPTIONAL cheap parallel de-risk (orchestrator CAN run, ~57 GB bf16 load, trough ~57 GiB ≫ 16,
SAFE): `--binding-dryrun --no-4bit --base models/qwen3-30b-wp-30_70-merged`. CAVEAT: bf16
binding ≠ 4-bit binding (Unsloth LoRA-attach onto bnb-4bit fused experts is a different path);
this is a PARTIAL signal only, NOT a substitute for step 3's 4-bit gate.

---
### UPDATE 2026-06-04 ~23:10 — memory blocker DIAGNOSED (see output/.../MEMORY-INVESTIGATION.md)

Ran 3 watchdog'd dry-runs. **Memory cap is HARMFUL, not protective** (pre-allocs ≈ the cap on
unified memory). Default now `DEFAULT_MAX_MEMORY_GIB = 0` (no cap). Real blocker: base
`merged-v2` is **57 GB bf16**; Unsloth's on-the-fly 4-bit load holds the bf16 shards →
**~100 GiB peak working set** (genuine — `expandable_segments` changed it < 0.2 GiB). Full load
needs ~108 GiB peak vs 121 w/ 16 floor → unsafe on live desktop. Watchdog
(`scripts/_binding_dryrun_watchdog.sh`, trip @20 GiB) tripped cleanly 3×, NO host reboot.

**Binding guard PRIMARY path independently validated** on a CPU toy MoE
(`scripts/_toy_moe_binding_check.py`): expert-LoRA→delta 2.82→BOUND; attn-only→0.0→BINDING_FAILED.
Probe logic correct regardless of the 30B load.

### UPDATE 2026-06-04 ~23:25 — peak is INTRINSIC; path = multi-user.target pre-quant (USER chose)

Tested transformers-native `BitsAndBytesConfig` 4-bit load too: ALSO peaks ~100.6 GiB
(min_avail 20.37) — IDENTICAL to Unsloth & expandable_segments. The peak is intrinsic to
unified memory (CPU+GPU one pool → bf16 staging + growing 4-bit copy stack), loader-independent.
All 3 runs bottom at a reproducible ~20.3 GiB then recover. **USER chose multi-user.target**
(safest: no GNOME OOM-victim + ~6 GiB headroom). NOTE: this session is x11/tty2 + sudo needs a
password → orchestrator CANNOT do the isolate (privileged AND would kill its own session).

**ACTION — user runs the detached root job (then re-resumes):**
```
sudo systemd-run --unit=prequant-mergedv2 --collect bash \
  /home/robert_li/Desktop/projects/wp-finetune/scripts/_prequant_multiuser.sh
# watch from another TTY: journalctl -u prequant-mergedv2 -f ; tail -f logs/prequant_mergedv2.log
```
It isolates multi-user.target → pre-quant `merged-v2`→`models/qwen3-30b-wp-30_70-merged-v2-4bit`
(`scripts/_tf_4bit_peak_probe --save`, watchdog trip 12 GiB) → restores graphical.target (trap,
even on failure). ~6-8 min; desktop drops then returns.

**POST-CREATION (orchestrator, after user re-resumes):**
1. Verify `models/qwen3-30b-wp-30_70-merged-v2-4bit` exists + **loads cheaply in Unsloth**
   (~16 GB peak) AND `load_adapter(checkpoint-72)` works on it — the arms use Unsloth
   `FastLanguageModel`, so a transformers-saved bnb-4bit ckpt MUST be Unsloth-loadable. RISK: if
   Unsloth can't load it, either load via transformers+peft in the probe, or pre-quant via Unsloth.
2. Point `--base` at the 4-bit ckpt; retry `--binding-dryrun` (now ~16 GB, safe on live desktop)
   → BOUND/BINDING_FAILED → **Task 1 gate passes**.
3. CP1 human-verify, then Task 2 (3 arms; the 2 Unsloth-4bit arms now load the 4-bit ckpt cheaply).

---
ARCHIVED earlier-step trail (superseded by the above):
1. ~~Cheap test: plain transformers `BitsAndBytesConfig` 4-bit load~~ — DONE, also peaks ~100 GiB.
2. **If yes → pre-quantize `merged-v2` to bnb-4bit ONCE** on live desktop (safe); then dryrun +
   both Task-2 4-bit arms load 4-bit directly at ~16 GB peak. UNBLOCKED.
3. **If it ALSO peaks ~100 GB →** bf16 materialization intrinsic; pre-quantize under
   `multi-user.target` (no GNOME victim, +6 GiB) OR hand creation to USER (debug-note: docker
   exec + watchdog + `journalctl -f -k | grep -iE 'oom|NV_ERR'`). ← USER decision (disrupts desktop).

Superseded original action (kept for trail): retry `--binding-dryrun --max-memory-gib 24` —
that cap fails fast (bnb CPU-spill reject); the cap approach is abandoned.

</details>

## Key Decisions

- Inline-on-main-tree execution (gitignored model/output dirs break worktree isolation).
- Binding guard PRIMARY = forward-ACTIVATION delta (not the plan's literal named_modules
  lora_A/lora_B, which false-fails a `target_parameters` parametrization) — intent-preserving
  deviation, recorded in `1acd70d`.
- Unsloth runs INSIDE `unsloth-headless` (workdir `/workspace/wp-finetune`); host conda has no
  torch accelerator on GB10. vLLM arm runs via its own container (serve_30_70_vllm.sh).

## Critical Anti-Patterns

| Anti-pattern | Severity | Why |
|---|---|---|
| Running a 30B Unsloth load on the live desktop without a free-RAM watchdog | **blocking** | A 4-bit 30B load drained free RAM 115→16.5 GiB (NVRM pre-allocates ≈ the `max_memory` budget). Below ~16 GiB the kernel OOM-cascade kills the GNOME desktop / reboots the host (container python survives at oom_score_adj=0). ALWAYS watchdog `free` + `pkill` at ~18 GiB, and start well above the 60 GiB floor. Ref: `.planning/debug/resolved/parse-check-reboots-host.md`. |
| Running checkpoint_parse_check on host conda python | **blocking** | No torch accelerator on GB10 host → `NotImplementedError`. Must `docker exec unsloth-headless`. |
| `systemctl isolate multi-user.target` as a "memory fix" | warning | Frees only ~6 GiB (desktop uses ~6). Removes the OOM victim, not the cause. Not a fix. |
| Executing `04.3-REOPEN-PLAN.md` | warning | 0-task brief, folded into 04.3-02. Skip. |

## Boundaries (hold throughout)

ckpt-72 NOT promoted; `models/qwen3-30b-wp-30_70-reasoning-merged` + `...-merged-v2` (certified
fallback) are READ-ONLY; new artifacts only under `output/format_stability/discriminator/`.

## Resume

`/gsd:execute-phase 04.3` (only 04.3-03 incomplete). Re-read this file + STATE.md `stopped_at`.
Cheap parallel de-risk available: the binding-representation question can be answered on a toy
MoE on CPU (no 30B load) to validate the guard logic + test stub.
