"""Prototype for W1 collect_units: tree-sitter-php, no PHP runtime."""
import json, re, sys, time
from pathlib import Path
import tree_sitter_php as tsp
from tree_sitter import Language, Parser

PARSER = Parser(Language(tsp.language_php()))
SKIP_DIRS = {"vendor": "vendor", "node_modules": "node_modules", "tests": "test", "test": "test",
             "build": "build", "dist": "build", "bin": "test", "docs": "test"}
MAX_CHARS = 16000
OWNERS = {"class_declaration", "interface_declaration", "trait_declaration", "enum_declaration"}


def skip_reason(rel: str, text: str):
    parts = rel.lower().split("/")
    for p in parts[:-1]:
        if p in SKIP_DIRS:
            return SKIP_DIRS[p]
    name = parts[-1]
    if name.startswith("test-") or "-test." in name or rel.endswith(("Test.php", "Tests.php")):
        return "test"
    if name.endswith(".asset.php"):  # wp-scripts build output
        return "build"
    if re.search(r"extends\s+(WP_UnitTestCase|TestCase)\b", text):
        return "test"
    if name.endswith(".min.php") or is_minified(text):
        return "minified"
    return None


def is_minified(text: str) -> bool:
    lines = text.splitlines() or [""]
    return len(text) > 2000 and len(text) / len(lines) > 200
    # ponytail: line-length heuristic; misses long one-line arrays only if <2000 chars


def owner_name(node):
    p = node.parent
    while p is not None:
        if p.type in OWNERS:
            return p.child_by_field_name("name").text.decode()
        if p.type == "anonymous_class":
            return "class@anonymous"
        p = p.parent
    return None


def code_of(node, src: bytes):
    kw = next(c for c in node.children if c.type == "function")  # drop modifiers, like training data
    start = kw.start_byte
    prev = node.prev_named_sibling
    if prev is not None and prev.type == "comment" and prev.text.startswith(b"/**") \
            and node.start_point[0] - prev.end_point[0] <= 1:
        start = prev.start_byte
    return src[start:node.end_byte].decode("utf-8", "replace")


def truncate(code: str):
    if len(code) <= MAX_CHARS:
        return code, False
    cut = code[:MAX_CHARS].rsplit("\n", 1)[0]
    return cut + "\n    // ... [truncated by wpjudge]\n", True


def file_units(rel: str, src: bytes):
    tree = PARSER.parse(src)
    units, skips, stack = [], [], [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type in ("function_definition", "method_declaration"):
            body = n.child_by_field_name("body")
            if body is not None:  # abstract / interface methods have no body
                fn = n.child_by_field_name("name").text.decode()
                own = owner_name(n)
                name = f"{own}::{fn}" if own else fn
                line, end = n.start_point[0] + 1, n.end_point[0] + 1
                if n.has_error:
                    skips.append({"file": rel, "line": line, "name": name, "reason": "parse_error"})
                elif end - line + 1 < 3:
                    skips.append({"file": rel, "line": line, "name": name, "reason": "tiny"})
                else:
                    code, tr = truncate(code_of(n, src))
                    units.append({"id": f"{rel}:{line}:{name}", "file": rel, "line": line, "end_line": end,
                                  "name": name, "kind": "method" if own else "function",
                                  "code": code, "loc": end - line + 1, "truncated": tr})
        stack.extend(reversed(n.children))
    if not units and not any(s["reason"] == "tiny" for s in skips):
        t = template_unit(rel, src, tree)
        if t:
            (units if t.get("kind") else skips).append(t)
    return units, skips


def template_unit(rel, src, tree):
    """Files with top-level PHP statements and no functions/classes -> one template unit."""
    stmts = [c for c in tree.root_node.named_children if c.type not in ("php_tag", "text", "comment")]
    if not stmts or any(c.type in OWNERS for c in stmts):
        return None
    loc = src.count(b"\n") + 1
    if loc < 3:
        return {"file": rel, "line": 1, "name": "<template>", "reason": "tiny"}
    code = src.decode("utf-8", "replace")
    if len(code) > MAX_CHARS:
        return {"file": rel, "line": 1, "name": "<template>", "reason": "too_large"}
    return {"id": f"{rel}:1:<template>", "file": rel, "line": 1, "end_line": loc, "name": "<template>",
            "kind": "template", "code": code, "loc": loc, "truncated": False}


def collect_units(root):
    root = Path(root)
    units, skips = [], []
    for p in sorted(root.rglob("*.php")):
        if p.is_symlink() or not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        src = p.read_bytes()
        r = skip_reason(rel, src.decode("utf-8", "replace"))
        if r:
            skips.append({"file": rel, "line": 0, "name": "", "reason": r})
            continue
        u, s = file_units(rel, src)
        units += u
        skips += s
    return units, skips


if __name__ == "__main__":
    t0 = time.perf_counter()
    units, skips = collect_units(sys.argv[1])
    dt = time.perf_counter() - t0
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps({"units": units, "skips": skips}))
    from collections import Counter
    print(json.dumps({"secs": round(dt, 2), "kinds": Counter(u["kind"] for u in units),
                      "truncated": sum(u["truncated"] for u in units),
                      "skips": Counter(s["reason"] for s in skips)}))
