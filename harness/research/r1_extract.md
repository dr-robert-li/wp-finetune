# R1 — Extraction: target → `Unit` / `Skip`

Host: aarch64, Python 3.12, no php, no docker access. Prototypes: `$S=.../scratchpad/r1/tools/{ts_extract,intake}.py`.

## Options

| | (a) fix `php_extract_functions.php` | (b) tree-sitter-php (pip) | (c) nikic/php-parser |
|---|---|---|---|
| Runtime | php-cli 8.x (host: none; static binary 13 MB from dl.static-php.dev works without sudo) | `tree-sitter 0.26.0` + `tree-sitter-php 0.25.1`, abi3 **manylinux aarch64 wheels**, 4.6 MB, no compiler | php-cli + composer vendor dir |
| Known bugs | all 5 present (below), plus traits, attributes, method named `print` | none found | none, but heaviest |
| Speed | ~0.19 s/file (process per file): cf7 83 files 15.7 s; woocommerce projected ~11 min | woocommerce 3,664 files **2.5 s** (107 MB RSS); cf7 0.09 s | n/a |
| Image / host cost | php-cli in image, `sudo apt install php-cli` on hosts | two pip deps | worst |

(c) Regex or brace matching is the same broken idea as (a), and there is no stdlib PHP parser.

## Evidence

**Edge-case file** (`$S/edge/edge.php`): braced `namespace Foo {}` + `namespace {}`, `"{$x->id}"` and `"${x}"`, an interface with 2 bodiless methods, an abstract class (abstract method + `after()`), `new class { inner() }`, a closure inside `Real::m`, `global_fn` in the global namespace.

Original script (`$S/php/php scripts/php_extract_functions.php edge.php`):
```
3  2 curly          <- ends at line 4 ({$ not counted)
9  8 Base::area     <- interface method swallows the next code, wrong owner
18 8 make           <- Base::after, abstract build, anon-class inner all lost
27 4 Real::m
34 3 Real::global_fn <- class context leaks across namespaces
```
tree-sitter prototype (line, end_line, kind, name):
```
3 7 function curly          14 16 method Base::after
18 25 function make         20 22 method class@anonymous::inner
27 30 method Real::m        34 36 function global_fn
```
Bodiless methods are dropped and the closure stays inside `Real::m`. All correct.

**Real targets** (wordpress.org latest-stable zips):

| target | .php files | functions | methods | templates | skips | truncated | time |
|---|---|---|---|---|---|---|---|
| classic-editor | 1 | 0 | 39 | 0 | tiny 1 | 0 | 0.01 s |
| contact-form-7 | 159 | 317 | 436 | 6 | vendor 70, build 6, tiny 15 | 0 | 0.09 s |
| twentytwentyfour | 58 | 3 | 0 | 57 | – | 0 | 0.01 s |
| woocommerce | 3,664 | 1,027 | 18,756 | 368 | vendor 520, build 307, tiny 87, too_large 8, minified 1 | 22 | 2.5 s |

Every unit has `end_line ≥ line+2`. In woocommerce, unit code length is p50 550, p99 6,323 and max 16k chars. No `parse_error` occurred (tree-sitter tolerates errors; units whose subtree `has_error` are skipped).

**Head-to-head on contact-form-7:** the original script disagrees with tree-sitter in **16 of 83 files**. The causes are traits (methods lose `Class::`), `{$…}` ending functions early (`construct_properties` 277–316 vs the true 277–326), interface methods swallowing the next one (`get_title` 174–180), and methods named `print` being missed.

**Token budget:** measured with the v4 tokenizer on 3,307 units over 1,500 chars. Chars per token: p50 3.39, p5 2.96, **p1 2.72**, min 1.53 (non-ASCII strings).

## Recommendation: (b) tree-sitter-php, one module `wpjudge/extract.py`

- **Walk:** `function_definition` and `method_declaration` nodes that have a `body`. The owner is the nearest `class/interface/trait/enum_declaration` name, or `class@anonymous`. Name: `Owner::fn` or `fn`.
- **Code** = the adjacent `/** docblock */` + the source from the `function` keyword to the node end. This matches training data (84/141 val samples start `/**`, the rest `function `; none `public`). `line` and `end_line` come from the node.
- **Skip rules** (one `Skip` per file, `line: 0, name: ""`):
  - vendor: path segment `vendor`.
  - node_modules: path segment `node_modules`.
  - build: path segment `build` or `dist`, or the name ends in `*.asset.php` (wp-scripts output; 6 in cf7, 307 in woo).
  - test: path segment `test`, `tests`, `bin` or `docs`; or the name starts with `test-`, contains `-test.`, or ends with `Test.php` / `Tests.php` (case-sensitive); or the file matches `extends (WP_UnitTestCase|TestCase)`.
  - Per unit: tiny if `end_line-line+1 < 3`; parse_error if the subtree `has_error`.
  - Deviations from `has_test_code` (woocommerce false positives): no bare `PHPUnit` substring (matched a comment); no lowercase `test.php` suffix (matched `abtest.php`).
- **Minified:** `name.endswith(".min.php")` or (`len > 2000` and average line length `> 200`). On woocommerce this caught 1 file (12 KB on one line). A max-line rule caught `MobileDetect.php` (long regex lines) and was dropped.
- **Template unit:** a file qualifies when it has no function or method units, no tiny skips, and no class/interface/trait/enum, but has at least one top-level statement other than `php_tag`, `text` or `comment`. It becomes one unit: `name "<template>"`, `kind "template"`, `line 1`. If it is over `MAX_CHARS`, skip it instead with reason **`too_large`** (new reason; contract amendment). Templates are never truncated.
- **Truncation:** the input budget is 8192 − 2048 (output) − ~50 (wrapper + chat template) ≈ 6,090 tokens. **`MAX_CHARS = 15000`** gives ≈ 5,500 tokens at the p1 ratio. Cut at the last newline, append `// ... [truncated by wpjudge]`, set `truncated: true`. Pure character counting, no tokenizer dependency. W2 safety net: on a context-overflow error, the retry halves the code.
- **Archive intake** (stdlib, tested), `$S/tools/intake.py`:
```python
def _check_name(name):
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
        raise Rejected(f"unsafe path: {name}")
def _check_totals(total, n, packed):
    if n > MAX_FILES: raise Rejected("too many files")                # 20000
    if total > MAX_TOTAL or (total > 10*2**20 and total > MAX_RATIO*packed):  # 200 MB, 100x
        raise Rejected("uncompressed size/ratio limit")
# zip: check every infolist() name; reject S_IFLNK in external_attr>>16; totals from file_size; extractall
#      (zipfile stops at the declared size + CRC, so lying headers cannot exceed it)
# tar: tarfile.open (auto gz); reject issym()/islnk(); totals from m.size; extractall(dest, filter="data")
# then: a single top-level dir -> root = that dir
```
  Tested: ok.zip and ok.tar.gz accepted. Rejected: zip `../../x`, `/tmp/x`, zip symlink, 300 MB bomb, 5 MB at ~1000×, tar `../x`, tar symlink, tar hardlink, non-archive. **No file escaped.** The 4 real zips (ratios 1.1–3.2) pass; woocommerce extracts in 0.31 s. In `collect_units`, also skip `p.is_symlink()` (covers directory input).

## W1 file list
- `harness/wpjudge/extract.py`: `load_target` (intake guards above, dir passthrough) + `collect_units` (walk, skips, minified, template, truncation). About 180 lines.
- `harness/pyproject.toml`: add `tree-sitter>=0.25,<0.27` and `tree-sitter-php>=0.24,<0.26`.
- `harness/tests/fixtures/edge.php` (the file above) + the check `harness/tests/test_extract.py`. It asserts the 6 edge units and lines, and that the malicious archives are rejected.
- **Drop:** `wpjudge/php/extract_functions.php`; php-cli in the image (amend SPEC R3.3); the php-cli host blocker in PLAN.

## Risks
- **Grammar lag** on new PHP syntax: `has_error` gives a `parse_error` skip, not wrong units. Pin the versions.
- **Templates dominate themes:** 57 of 60 units in twentytwentyfour are block patterns, mostly HTML with `esc_html_e`. They are out of distribution for the judge; add a `WPJUDGE_TEMPLATES=0` toggle if R2 shows junk.
- **Double counting:** an anonymous-class method inside a function is judged twice. Rare; accepted.
- **The char budget is a heuristic:** the min ratio was 1.53, so a 15k-char string-heavy unit could reach about 9.8k tokens. The W2 halve-on-overflow retry covers it.
- **The ratio guard trusts declared sizes**; zip enforces them on read, tar relies on `filter="data"`.
