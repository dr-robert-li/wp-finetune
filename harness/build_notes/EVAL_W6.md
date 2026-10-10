VERDICT: PASS

Evaluator: independent, W6 (fixtures + e2e script), re-evaluation after fix round 1. Basis: PLAN W6 entry, phase 5 list, binding changes 1, 2, 3, 7, 8, 12-15; build_notes/W6.md "Fix round 1"; round-1 EVAL_W6.md.
All eight round-1 defects are fixed and no regressions were found. No new defects of major or blocker severity.

## Defects
1. **minor (observation, no fix required)**, `tests/e2e.sh:176-180` (chk.py `cmd_golden`). The planted < clean check skips pairs with a null overall, so an all-unjudged run makes it vacuous.
   - This is already covered: the same run fails the `coverage >= 90` assert at line 162, and the SQLi-cap and anti_patterns asserts also fail. I probed this; see below.
   - Listed only so a future reader does not count that line as independent evidence.

## Round-1 defects re-checked
1. **major, fixed.** `e2e.sh:174` now iterates `R.items()` (dir, zip, tgz) only for the cap, planted < clean and anti_patterns asserts.
   - w4 still gets schema, fixture_units and recompute (line 159), plus verdict equality against dir (line 169), which is what binding change 3 asks for.
   - Probe: a w4 report with the SQLi units' security and sql_safety drifted from 4 to 5 gives 0 FAILs. A w4 report with the SQLi verdicts flipped gives exactly one FAIL, `workers=4 vs workers=1: verdicts equal`.
2. **fixed.** `fixture_units` (line 148) asserts `files_scanned == exp["php_files"]`, and it is used by fixture, golden and serverdown. Probe: a dir report with files_scanned 6 FAILs `dir: files_scanned 6 == 7`.
3. **fixed.** Line 166 uses `max([...], default=0)`. Line 169 uses `b.get(i, {}).get("verdict")` and is guarded by `bool(a)`.
   - Probe: an all-unjudged zip report plus a w4 report missing units raised no Traceback. The degenerate inputs surface as FAIL lines.
4. **fixed.** Line 177 is the strict `<`. Probe: planted SQLi overall 99 FAILs with the violating pairs listed.
5. **fixed.** `abs.tar.gz` (line 284) is in the set and in the section loop (line 313). The live run rejected it with exit 2 and `unsafe path: /tmp/wpjudge_e2e_escape_tabs.php`.
6. **fixed.** Line 483 is `find / /data/jobs -xdev`. I verified that GNU find applies `-xdev` per starting point: `find / /dev/shm -xdev` does list files in the separate tmpfs. So the jobs volume is searched.
7. **fixed.** Line 463 is now `fail` when no partial `/functions` read is observed.
8. **fixed.** `simple-events.php:21` is now `add_filter('sanitize_option_ser_settings', 'ser_sanitize_settings')`. The callback exists at line 99, and line numbers are unchanged. `collect_units` still equals `expected.json` (10 units, files_scanned 7 == php_files 7). The prerun ids still match.

## What I ran
- `bash -n tests/e2e.sh`: OK.
- `WORK=<scratch> bash tests/e2e.sh malicious serverdown`: **35 passed, 0 failed**, 4.8 s.
  - All 13 malicious archives exit 2 with no traceback and no output.
  - Temp dirs are cleaned and nothing escaped.
  - The serverdown report is valid.
- `collect_units(fixture)` vs `expected.json`: identical.
- Golden mutation probes with the generated `chk.py`, using synthetic reports built with `aggregate.build_report` from the prerun scores:
  - baseline: 0 FAIL
  - w4 drift: 0 FAIL
  - w4 verdict flip: 1 FAIL
  - planted above clean: 3 FAIL (strict order, plus dir vs zip/tgz)
  - files_scanned wrong: 2 FAIL
  - all-unjudged zip plus w4 missing units: FAILs, no crash
- Not run, because they need the GPU, a model server or docker: cli, library, api, allinone, parity, real, prerun.

---
Round 1: FAIL (1 major: the workers=4 run got score/cap assertions against binding change 3; 7 minor). All fixed in fix round 1.
