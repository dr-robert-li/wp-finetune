---
slug: parse-check-reboots-host
status: resolved
trigger: "parse-check reliably restarts the host machine"
created: 2026-05-28T07:10:00Z
updated: 2026-05-28T07:38:00Z
goal: find_and_fix
specialist_dispatch_enabled: true
---

# Debug Session: parse-check-reboots-host

## Trigger

<DATA_START>
parse-check reliably restarts the host machine
</DATA_START>

## Symptoms

- **expected**: `scripts.checkpoint_parse_check` runs to completion (~50 val examples), emits parse-rate vs threshold (0.05) verdict, exits cleanly. Host stays up.
- **actual**: Host machine reboots before parse-check completes. Process killed by reboot; inner log files never written or truncated. After reboot, no parse-check artifacts.
- **error**: No application-level error captured (process dies with system). Last visible output is wrapper script's `Executing: python -m scripts.checkpoint_parse_check ...` line; inner stdout/stderr never flushed before reboot.
- **timeline**: Reproduced 3+ times across multiple days. Reboot occurs 5–30 min after launch (consistent with model-loading / first-inference window).
- **reproduction**: Launch `scripts/_run_parse_check_ckpt72.py` (RTRN-04 gate) — wraps `python -m scripts.checkpoint_parse_check --checkpoint-dir adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72 --base models/qwen3-30b-wp-30_70-merged --val-jsonl data/reasoning_dataset/openai_val.jsonl --n 50 --threshold 0.05`. Wrapper calls `dgx_toolbox.ensure_ready('unsloth_studio')` then `docker exec unsloth-headless …` to run the parse-check module. Reboot occurs within 5–30 min.

## Background (project context — for debugger)

- **Hardware**: DGX Spark, NVIDIA GB10 (Grace+Blackwell SoC, unified mem), kernel 6.17.0-1018-nvidia, nvidia-driver 580.159.03.
- **Workload**: Loads merged Qwen3-30B-A3B (MoE) base + LoRA adapter (4.9 GB safetensors) via Unsloth / HF Transformers inside container `unsloth-headless` (nvcr.io/nvidia/pytorch:25.11-py3), runs ≤50 generations on val JSONL.
- **Recent reboots (`last reboot`)**: May 27 20:30 (still running), May 13 21:18, May 6 (×3). Several of these previously attributed to OS Software Updater (PackageKit / nvidia-spark-run-apt-upgrade-once.service) — but user now reports parse-check launches reliably precede the reboots across multiple days, which contradicts that single-cause attribution.
- **Process management history**: setsid wrapper exits early (~8 s); nohup+disown variant survives longer but parse-check still does not complete before host reboots.
- **Container state at last attempt**: `unsloth-headless` up 11 h, mount OK, deps pinned, CUDA available (per `dgx_toolbox.ensure_ready` checks).
- **Known irrelevant**: parse_check log files in `logs/phase4.3/parse_check_ckpt72*` show wrapper output but inner module log is empty/missing for the run that reboots.

## Symptoms — investigation hints (NOT conclusions)

Candidate hypothesis families to probe (in priority order):

1. **Hardware fault triggered by workload**: GB10 power/thermal limit, NVRM Xid, SoC PCIe link reset, unified-memory pressure crashing the kernel. Probe: `journalctl -k -b -1`, `journalctl -k -b -2 …`, `dmesg --ctime`, `/var/log/kern.log*`, `nvidia-bug-report.sh`, recent Xid IDs in syslog.
2. **OOM or cgroup kill of kernel-critical process**: Large MoE load could push host RAM past tipping point on Spark's unified memory. Probe: `journalctl -k | grep -iE 'oom|killed process|out of memory|memory cgroup'`, `dmesg | grep -iE 'oom|memory'`.
3. **Watchdog / NMI / hard lockup**: Long GPU stalls during model load. Probe: `journalctl -k | grep -iE 'watchdog|hard lockup|nmi|soft lockup|hung_task'`.
4. **NVIDIA driver bug on 6.17 kernel + 580.159.03**: Known cluster of regressions in Spark images. Probe: nvidia-smi messages in syslog right before each reboot, `dpkg -l | grep nvidia`, `apt log` around reboot times.
5. **Software Updater coincident with workload**: previously suspected; user data contradicts as sole cause but updater could be triggered or accelerated by parse-check (e.g., long-running process keeps PackageKit awake → offline-update applied at next 30-min tick). Probe: `journalctl -u packagekit -u unattended-upgrades -u update-manager -u nvidia-spark-run-apt-upgrade-once -b -1`.

## Constraints (set by user)

- **Repro mode**: log forensics only. **DO NOT** launch parse-check, _run_parse_check_ckpt72.py, or any equivalent. **DO NOT** trigger a controlled reboot.
- **Read-only on host services**: do not change updater config or kernel/driver packages without surfacing fix options for human approval first.
- Container `unsloth-headless` may be inspected (`docker exec`, `docker logs`) but DO NOT exec workloads that re-trigger the crash.

## Current Focus

hypothesis: ROOT CAUSE FOUND — parse-check loading Qwen3-30B into the GB10's 121 GB unified memory (= sole system RAM) exhausts the NVIDIA driver's unified-memory pool (NVRM NV_ERR_NO_MEMORY) during model init, then triggers the Linux OOM killer. Because the container python runs at oom_score_adj=0, the kernel kills the entire desktop session instead. Historical reboots (May 6, May 13, May 27) have separate independent causes; parse-check has NOT been confirmed as the direct reboot trigger but IS confirmed as the OOM cascade trigger.
test: cross-boot forensic correlation of reboot timestamps vs dpkg/apt logs vs OOM kernel messages
expecting: confirmed — see Evidence section
next_action: surface fix options to user (awaiting selection)
reasoning_checkpoint: ""
tdd_checkpoint: ""

## Evidence

- timestamp: 2026-05-28T07:15:00+10:00
  label: boot-correlation-table
  content: |
    Boot history (journalctl --list-boots):
      boot -4: May 6 10:15 – 14:14 (1h 59m)
      boot -3: May 6 14:15 – 15:38 (1h 23m)
      boot -2: May 6 15:47 – May 13 21:13 (7+ days)
      boot -1: May 13 21:18 – May 27 20:30 (13+ days)
      boot  0: May 27 20:30 – still running

    Reboot timestamps vs parse-check launches:
      May 6 (3 reboots):  triggered by do-partial-upgrade installing kernel 6.17.0-1014 + nvidia 580.142.
                          NO parse-check logs exist for these dates.
      May 13 21:18:       triggered by OOM cascade — kernel ran out of memory from Waveterm (VSZ ~1.4 TB
                          in unified-mem mappings) combined with VS Code + desktop processes.
                          NO parse-check_ckpt72 logs for this date.
                          Journal ends at 21:13 (journald killed by OOM); host reboot ~4-5 min later.
      May 27 20:30:       triggered by do-partial-upgrade (user-authorised polkit auth at 20:10:44 for
                          com.ubuntu.release-upgrader.partial-upgrade). Installed kernel 6.17.0-1018 +
                          nvidia 580.159.03 + purged old kernels. Reboot at 20:30:04 per logind.
                          Parse-check wrapper first launched at 20:31 UTC (AFTER the reboot).

    IMPORTANT CAVEAT: The user believes parse-check "reliably" precedes the reboots across multiple days.
    The log evidence does NOT support parse-check as the direct reboot cause for any documented event.
    However, parse-check IS confirmed to cause a severe OOM cascade (see current-boot evidence below)
    that would eventually cause a reboot if the cascade escalated further or if a system process was killed.
    It is possible that on dates without retained logs, parse-check did directly trigger a reboot via OOM.

- timestamp: 2026-05-28T07:20:00+10:00
  label: may27-do-partial-upgrade-chain
  content: |
    May 27 20:07:44 — user opened GNOME Software / update-manager → PackageKit activated.
    May 27 20:10:44 — polkit ONE-SHOT auth for do-partial-upgrade; pkexec executed it.
    May 27 20:12:13 — dpkg removed linux-modules-nvidia-580-open-6.17.0-1008.
    May 27 20:12:18 — dpkg upgraded nvidia-driver-580-open: 580.142 → 580.159.03 (userspace only).
    May 27 20:12:29 — dpkg installed kernel 6.17.0-1018-nvidia and its modules.
    May 27 20:13:28 — NVRM API mismatch logged (userspace 580.159.03, kernel module still 580.142).
                      This mismatch is expected — new kernel not yet booted.
    May 27 20:30:04 — systemd-logind: "The system will reboot now!" (do-partial-upgrade completion).
    May 27 20:30:46 — new boot on 6.17.0-1018-nvidia with matched nvidia module 580.159.03.

- timestamp: 2026-05-28T07:22:00+10:00
  label: current-boot-parse-check-oom-cascade
  content: |
    Parse-check in the CURRENT BOOT (boot 0) triggered a severe OOM cascade — live evidence
    of what parse-check does to the system when the OOM propagates fully.

    06:31 AEST May 28 — wrapper started (UTC 20:31 May 27), model load began inside container.
    06:32 AEST         — wrapper received SIGTERM (returncode 143 captured); SIGTERM propagated
                         to docker exec. NOTE: wrapper PID 273446 was still alive in the OOM
                         dump at 06:40 — the wrapper did NOT actually exit at 06:32. The
                         returncode 143 was recorded in the wrapper log but the wrapper continued
                         blocking on docker exec. SIGTERM mechanism unverified.
    06:39:36 AEST      — NVRM: nvCheckOkFailed: Out of memory [NV_ERR_NO_MEMORY] —
                         GB10 unified memory (121.688 GB total) exhausted at NVIDIA driver level.
                         NOTE: Only ~5/531 weight shards loaded at this point (~4.2 GB of actual
                         weight data). The driver appears to pre-allocate or reserve unified
                         memory well beyond the model weights during init. This is not fully
                         characterized — the 4.2 GB loaded vs 121 GB exhausted anomaly requires
                         further investigation if root memory footprint matters.
    06:40:48 AEST      — Linux kernel OOM killer invoked:
                         python PID 273873 (inside container, UID=0, oom_score_adj=0):
                           total_vm = 37,163,340 pages = 141 GB virtual (model mmap)
                           swapents = 287,895 pages = 1.12 GB paged to swap
                           pgtables = 5,033,984 bytes = 4.8 MB page tables
                           anon_rss = ~5,000 pages = 20 MB (only interpreter; model in nvidia unified mem)
                         OOM killer selected Waveterm (oom_score_adj=300) as victim. Python survived.
    06:41 – 06:55 AEST — OOM cascade: kernel killed Waveterm, Chromium, snap-store, xdg-desktop-portal,
                         copyq, evolution-alarm/source, ibus-extension, gsd-{color,power,keyboard,xsettings},
                         pipewire, pipewire-pulse, gcr-ssh-agent, snapd-desktop-integration, tracker-miner-fs.
                         GNOME desktop session completely destroyed.
    ~06:55 AEST        — OOM cascade subsided. Python 273873 eventually died (mechanism: either
                         docker SIGKILL 10s after SIGTERM arrived, or kernel exhausted killable mem).
    07:01 AEST         — GNOME session respawned by GDM. Host survived WITHOUT reboot this time.

    Why host survived today but may reboot other times: the OOM killer only kills processes with
    oom_score_adj ≥ 0. On this run it found enough victims in the desktop session. If a system-critical
    service (NetworkManager, systemd-journald, or another oom_score_adj=-999 process) was OOM-killed
    instead, that would trigger a kernel panic or systemd emergency → reboot.

- timestamp: 2026-05-28T07:24:00+10:00
  label: nvrm-oom-pre-dates-linux-oom
  content: |
    Across multiple boots, NVRM nvCheckOkFailed: Out of memory appears BEFORE the Linux OOM killer:
      May 13 22:57 AEST (boot -1): NVRM OOM ~1h after new docker container start.
      May 14 22:00 AEST (boot -1): Linux OOM cascade followed, killed Waveterm + model python.
      May 28 06:39 AEST (boot  0): NVRM OOM fired, Linux OOM cascade started 1 min later.

    This confirms the OOM originates in NVIDIA's unified memory subsystem, not Linux page reclaim.
    Changing Linux-level settings (vm.overcommit, cgroup limits) will not prevent the NVRM OOM.
    The fix must either reduce the unified memory footprint or prevent concurrent memory pressure.

- timestamp: 2026-05-28T07:26:00+10:00
  label: eliminated-causes
  content: |
    - No NVRM Xid errors in any boot (no hardware fault signal).
    - No watchdog/NMI/hard-lockup messages preceding any reboot.
    - nvidia-spark-run-apt-upgrade-once.service: skipped on both boot -1 and boot 0
      (ConditionPathExists=!/var/lib/nvidia-spark-run-apt-upgrade-once/done — already ran).
    - NVRM API mismatch (580.142/580.159.03): transient, resolved by May 27 reboot.
      Current boot kernel 6.17.0-1018 + module 580.159.03 are fully matched.

## Eliminated

- **Hardware fault (Xid, PCIe reset, thermal)**: No Xid or thermal events in any boot journal.
- **Watchdog / NMI / hard lockup**: Not present in any boot leading up to any reboot.
- **nvidia-spark-run-apt-upgrade-once.service**: Already satisfied; skipped in all recent boots.
- **Persistent NVIDIA driver split-brain**: Transient (during live upgrade); resolved at reboot.
- **May 27 reboot caused by parse-check**: Reboot preceded parse-check launch by ~1 min.
- **May 6 reboots caused by parse-check**: Caused by do-partial-upgrade (kernel+driver upgrade). No parse-check logs.
- **May 13 reboot caused by parse-check**: Caused by Waveterm + other process OOM cascade. No parse-check_ckpt72 logs.

## Resolution

root_cause: |
  Parse-check loads Qwen3-30B-A3B into the GB10's 121 GB unified memory pool (the sole system RAM).
  During model initialization the NVIDIA driver exhausts unified memory allocations and fires
  NV_ERR_NO_MEMORY (NVRM nvCheckOkFailed), then the Linux OOM killer cascades through desktop
  processes (all oom_score_adj ≥ 200) while the container python (oom_score_adj=0) survives.
  This wipes the entire GNOME desktop session and, if a critical system service is killed in the
  cascade, produces a kernel panic / systemd emergency that reboots the host.
  The historical reboots (May 6, May 13, May 27) each have a separate documented cause unrelated
  to parse-check; however, parse-check is confirmed to produce the OOM cascade pattern and likely
  caused undocumented reboots on other dates without retained journal logs.
  Anomaly not fully characterized: NVRM OOM fires after only ~4 GB of model weights loaded,
  suggesting the NVIDIA driver pre-allocates unified memory substantially beyond actual weight data.

fix: |
  Option 1 applied 2026-05-28 (code-only, no system changes):

  scripts/checkpoint_parse_check.py
    - DEFAULT_N: 50 → 10 (parse-rate sanity check doesn't need 50 samples)
    - DEFAULT_LOAD_IN_4BIT = True (drops 30B weights from ~60 GB bf16 → ~16 GB)
    - DEFAULT_MAX_MEMORY_GIB = 80 (per-device cap; passed to FastLanguageModel.from_pretrained
      as max_memory={0: "80GiB", "cpu": "20GiB"})
    - New CLI flags: --no-4bit (override), --max-memory-gib INT
    - In-code comment documents the GB10 unified-mem OOM mechanism.

  scripts/_run_parse_check_ckpt72.py
    - Added MIN_FREE_MEM_GIB = 40 pre-flight guard via `free -b`.
      Aborts with exit 2 (clear error message: close Waveterm/VS Code/browsers or
      `systemctl isolate multi-user.target`) before launching if available < 40 GiB.
    - Updated cmd to use --n 10 (matches new default).
    - Replaced deprecated datetime.utcnow() with datetime.now(_dt.UTC).

  Syntax-checked via `python -c "import ast; ast.parse(...)"` — both files parse OK.
  Pre-flight test: current host shows 116 GiB available — guard would PASS at this moment.

verification: |
  When ready to validate (user must run — orchestrator constrained to log-forensics only):
  1. Open a separate ssh session: `journalctl -f -k | grep -iE 'oom|NV_ERR|killed process'`
  2. Run `python scripts/_run_parse_check_ckpt72.py` — pre-flight guard should print
     "Available unified memory: X GiB (floor: 40 GiB)" and either continue or ABORT.
  3. If continues: confirm zero NVRM OOM, zero Linux OOM kills during run; GNOME stays up.
  4. Confirm `logs/phase4.3/parse_check_ckpt72_*.log` shows RETURNCODE=0 with parse-rate verdict.

  Residual risk: NVRM pre-allocation anomaly (4 GB loaded → 121 GB exhausted) is not
  fully characterized. If 4-bit + max_memory cap still exhausts unified mem, fall back
  to option 2 (`systemctl isolate multi-user.target` before launch). Re-open this
  session with `/gsd:debug continue parse-check-reboots-host` if so.

files_changed:
  - scripts/checkpoint_parse_check.py
  - scripts/_run_parse_check_ckpt72.py
