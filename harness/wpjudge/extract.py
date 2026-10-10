"""Target -> Units/Skips. tree-sitter-php, no PHP runtime (SPEC A1, research/r1_extract.md).

load_target(path, dest=None, cfg=None) -> (root_dir, target_meta)
collect_units(root_dir, cfg=None)      -> (units, skips, files_scanned)
"""
import hashlib
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import tree_sitter_php as tsp
from tree_sitter import Language, Parser

from wpjudge.config import load

PARSER = Parser(Language(tsp.language_php()))
SKIP_DIRS = {"vendor": "vendor", "node_modules": "node_modules", "tests": "test", "test": "test",
             "build": "build", "dist": "build", "bin": "test", "docs": "test"}
OWNERS = {"class_declaration", "interface_declaration", "trait_declaration", "enum_declaration"}
MARKER = "\n    // ... [truncated by wpjudge]\n"


class Rejected(ValueError):
    """Bad target: not a dir/archive, unsafe member, link, too many files, too big, bomb."""


# ---------------------------------------------------------------- intake

def _check_name(name):
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
        raise Rejected(f"unsafe path: {name}")


def _check_totals(total, n, packed, cfg):
    if n > cfg.max_files:
        raise Rejected(f"too many files: {n} > {cfg.max_files}")
    if total > cfg.max_total_mb * 2**20:
        raise Rejected(f"uncompressed size {total} B over {cfg.max_total_mb} MB")
    if total > 10 * 2**20 and total > cfg.max_ratio * packed:
        raise Rejected(f"compression ratio over {cfg.max_ratio}x")


def _extract(archive, dest, cfg):
    packed = archive.stat().st_size
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            for i in infos:
                _check_name(i.filename)
                if (i.external_attr >> 16) & 0o170000 == 0o120000:
                    raise Rejected(f"symlink: {i.filename}")
            _check_totals(sum(i.file_size for i in infos), len(infos), packed, cfg)
            z.extractall(dest)  # zipfile stops at declared size + CRC, so lying headers cannot exceed totals
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:  # .tar, .tar.gz, .tgz
            n = total = 0
            for m in t:  # streamed: stop at the first bad header, never buffer a million members
                _check_name(m.name)
                if m.issym() or m.islnk():
                    raise Rejected(f"link: {m.name}")
                n, total = n + 1, total + m.size
                _check_totals(total, n, packed, cfg)
            t.extractall(dest, filter="data")  # second line of defence
    else:
        raise Rejected(f"not a directory, zip or tar archive: {archive.name}")


def load_target(path, dest=None, cfg=None):
    """Dir passes through. Archive is extracted into <dest>/src (dest: caller-owned dir; default a new
    mkdtemp the caller must remove). Raises Rejected (ValueError) on bad input."""
    cfg = cfg or load()
    path = Path(path)
    if path.is_dir():
        root, sha, name = path, "", path.resolve().name
    elif path.is_file():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        made = None if dest else tempfile.mkdtemp(prefix="wpjudge-")
        out = Path(dest or made) / "src"
        out.mkdir(parents=True, exist_ok=False)
        try:
            try:
                _extract(path, out, cfg)
            except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as e:
                raise Rejected(f"bad archive: {e}") from e
        except Rejected:
            shutil.rmtree(made or out, ignore_errors=True)
            raise
        shutil.rmtree(out / "__MACOSX", ignore_errors=True)  # macOS zip resource forks
        kids = list(out.iterdir())
        root = kids[0] if len(kids) == 1 and kids[0].is_dir() else out  # unwrap plugin-slug/
        name = root.name if root is not out else re.sub(r"\.(zip|tar\.gz|tgz|tar)$", "", path.name)
    else:
        raise Rejected(f"no such file or directory: {path}")
    style = root / "style.css"
    theme = style.is_file() and not style.is_symlink() and "Theme Name:" in style.read_text("utf-8", "replace")
    return root, {"name": name, "type": "theme" if theme else "plugin", "sha256": sha}


# ---------------------------------------------------------------- units

def skip_reason(rel, text):
    parts = rel.lower().split("/")
    for p in parts[:-1]:
        if p in SKIP_DIRS:
            return SKIP_DIRS[p]
    name = parts[-1]
    # has_test_code minus 2 woocommerce false positives: bare "PHPUnit", lowercase "test.php" suffix
    if name.startswith("test-") or "-test." in name or rel.endswith(("Test.php", "Tests.php")):
        return "test"
    if name.endswith(".asset.php"):  # wp-scripts build output
        return "build"
    if re.search(r"extends\s+(WP_UnitTestCase|TestCase)\b", text):
        return "test"
    lines = text.count("\n") + 1
    if name.endswith(".min.php") or (len(text) > 2000 and len(text) / lines > 200):
        return "minified"  # ponytail: avg-line-length heuristic; max-line rule flagged MobileDetect.php
    return None


def _owner(node):
    p = node.parent
    while p is not None:
        if p.type in OWNERS:
            return p.child_by_field_name("name").text.decode()
        if p.type == "anonymous_class":
            return "class@anonymous"
        p = p.parent
    return None


def _code(node, src):
    """Adjacent /** docblock */ + source from the `function` keyword; attributes and modifiers dropped, as in
    training data (370/370 docblocked samples go `*/\nfunction`)."""
    code = src[next(c for c in node.children if c.type == "function").start_byte:node.end_byte]
    prev = node.prev_named_sibling
    if prev is not None and prev.type == "comment" and prev.text.startswith(b"/**") \
            and node.start_point[0] - prev.end_point[0] <= 1:
        code = prev.text + b"\n" + code
    return code.decode("utf-8", "replace")


def _truncate(code, max_chars):
    if len(code) <= max_chars:
        return code, False
    return code[:max_chars].rsplit("\n", 1)[0] + MARKER, True


def _unit(rel, line, end, name, kind, code, truncated):
    return {"id": f"{rel}:{line}:{name}", "file": rel, "line": line, "end_line": end, "name": name,
            "kind": kind, "code": code, "loc": end - line + 1, "truncated": truncated}


FUNCS = ("function_definition", "method_declaration")


def _orphan_error(n):
    """An ERROR/missing node outside any function: the grammar lost a function we cannot report by name."""
    if not (n.type == "ERROR" or n.is_missing):
        return False
    while n.parent is not None:
        n = n.parent
        if n.type in FUNCS:
            return False
    return True


def file_units(rel, src, cfg):
    tree = PARSER.parse(src)
    units, skips, stack = [], [], [tree.root_node]
    owned = orphan = False
    while stack:
        n = stack.pop()
        owned = owned or n.type in OWNERS or n.type == "anonymous_class"
        orphan = orphan or _orphan_error(n)
        if n.type in FUNCS and n.child_by_field_name("body"):
            own = _owner(n)
            fn = n.child_by_field_name("name").text.decode()
            name = f"{own}::{fn}" if own else fn
            line, end = n.start_point[0] + 1, n.end_point[0] + 1
            if n.has_error:
                skips.append({"file": rel, "line": line, "name": name, "reason": "parse_error"})
            elif end - line + 1 < 3:
                skips.append({"file": rel, "line": line, "name": name, "reason": "tiny"})
            else:
                units.append(_unit(rel, line, end, name, "method" if own else "function",
                                   *_truncate(_code(n, src), cfg.max_chars)))
        stack.extend(reversed(n.children))
    tpl = cfg.templates and not units and not skips and not owned
    if tpl:
        units, skips = _template(rel, src, tree, cfg)
    if orphan and not (tpl and (units or skips)):
        skips.append({"file": rel, "line": 0, "name": "", "reason": "parse_error"})
    return units, skips


def _template(rel, src, tree, cfg):
    """No functions/classes anywhere but top-level PHP statements -> one template unit (never truncated)."""
    if not [c for c in tree.root_node.named_children if c.type not in ("php_tag", "text", "comment")]:
        return [], []
    loc = src.count(b"\n") + 1
    code = src.decode("utf-8", "replace")
    reason = ("parse_error" if tree.root_node.has_error else "tiny" if loc < 3
              else "too_large" if len(code) > cfg.max_chars else None)
    if reason:
        return [], [{"file": rel, "line": 1, "name": "<template>", "reason": reason}]
    return [_unit(rel, 1, loc, "<template>", "template", code, False)], []


def collect_units(root_dir, cfg=None):
    """Walk *.php (no symlinks followed or read). files_scanned = regular .php files walked, skipped included."""
    cfg = cfg or load()
    root = Path(root_dir)
    units, skips, scanned = [], [], 0
    paths = sorted(Path(d) / f for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(".php"))
    for p in paths:
        if p.is_symlink() or not p.is_file():
            continue
        scanned += 1
        rel = p.relative_to(root).as_posix()
        src = p.read_bytes()
        r = skip_reason(rel, src.decode("utf-8", "replace"))
        if r:
            skips.append({"file": rel, "line": 0, "name": "", "reason": r})
            continue
        u, s = file_units(rel, src, cfg)
        units += u
        skips += s
    return units, skips, scanned
