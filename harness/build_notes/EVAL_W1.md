VERDICT: PASS

# W1 extract: re-evaluation after fix round 1

I judged this against SPEC.md, PLAN.md (Contracts, the W1 Locked plan, Binding changes), EVAL_W1 round 1 and the "Fix round 1" section of W1.md.
All seven round-1 defects are fixed. I probed each one myself and did not rely only on the builder's tests.
I found no regressions. The contract shapes are unchanged: `load_target` returns `(root, meta)` and `collect_units` returns `(units, skips, files_scanned)`, and W4's `scan()` and the CLI/API callers still match.
The full suite passes.

## Prior defects: status
1. **major, docblock kept attributes/modifiers: FIXED** (`extract.py:136-144`).
   - Probe: `/** e */ #[Pure] final public static function &s()` gives `/** e */\nfunction &s() {`.
   - A multi-line `#[A(\n 1\n)]` between the docblock and the method still attaches the docblock, because the method node starts at the attribute.
   - A docblocked plain function with `#[A]` gives `/** d */\nfunction g()`.
   - An abstract method with no body is still not emitted.
2. **major, tar guards after `getmembers()`: FIXED** (`extract.py:59-68`). I ran each probe in a fresh process.
   - The 1M-member tgz is rejected with "too many files: 20001" in 0.33 s at 28 MB RSS. In round 1 it took 15.7 s and 442 MB.
   - The 4 GB single-member tgz is rejected on its first header in 0.00 s.
   - The test (`getmembers` patched to fail, `next` counted) is real.
3. **minor, `__MACOSX` blocked unwrap: FIXED** (`extract.py:93`). Both zip and tgz with `__MACOSX/` unwrap to `slug`, and they leave no junk skip.
4. **minor, nested owners became templates: FIXED** (`extract.py:178,193`). The braced-namespace interface and the conditional class give no unit.
5. **minor, silent drop of unparseable function: FIXED** (`extract.py:161-169,196-197`).
   - `function bad( {` gives one per-file `parse_error` (line 0), and `ok()` is still emitted.
   - A bad class property (`public $x = ;`) gives the same per-file skip, and `K::m` is still emitted.
6. **minor, `.PHP` case: FIXED** (`extract.py:219`).
7. **minor, mkdtemp leak on Rejected: FIXED** (`extract.py:90-92`). A slip zip with `dest=None` leaves no `wpjudge-*` dir. See defect 1 below for the non-Rejected paths.

## Defects
1. **minor**, `wpjudge/extract.py:88`: encrypted zips and zips with an unsupported compression method are not wrapped as Rejected.
   - An encrypted zip raises `RuntimeError` ("password required"). A deflate64 zip (method 9) raises `NotImplementedError`.
   - Effects:
     - the CLI prints a traceback and exits 1, instead of `wpjudge: ...` and exit 2
     - with `dest=None`, the mkdtemp dir leaks (probe: two `wpjudge-*` dirs left)
     - the API is fine: it catches `Exception`, and `TemporaryDirectory` cleans up
   - This was there before round 1; it is not a regression.
   - Fix: add `RuntimeError` to the except tuple. It also covers `NotImplementedError`, which is a subclass.
2. **minor (note)**: memory use of the zip central directory is unchanged. Round 1 judged it tolerable, and I agree.

No over-engineering found. The file is 233 lines, and the fixes are small and local.

## What I ran
- `.venv/bin/python -m pytest tests -q`: 35 passed (10 in test_extract.py).
- Scratchpad probes (`scratchpad/w1r/p.py`, `t.py`, `m.py`):
  - docblock, attribute and modifier variants
  - orphan parse errors, at top level and in a class body
  - an anonymous class plus top-level code
  - an encrypted zip, a deflate64 zip and a corrupt-deflate zip (the last one is correctly Rejected)
  - the 1M-member tgz and the 4 GB tgz, each timed with RSS in a fresh process
  - `__MACOSX` in zip and tgz
- A stray `/tmp/wpjudge-mtlburp1` (empty `src/`, 10:42) exists, but the test suite does not create it (I checked on a re-run) and my probes do not either. Its origin is unknown, maybe another session. I left it alone.

---
Round 1 result: FAIL (2 major: docblock kept modifiers; tar guards ran after getmembers; plus 5 minor). All fixed in round 1.
