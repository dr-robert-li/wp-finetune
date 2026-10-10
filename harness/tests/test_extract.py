import io
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

from wpjudge.config import load
from wpjudge.extract import Rejected, collect_units, load_target

FIX = Path(__file__).parent / "fixtures" / "extract"


def test_edge_units():
    units, skips, scanned = collect_units(FIX)
    got = [(u["line"], u["end_line"], u["kind"], u["name"]) for u in units]
    assert sorted(got) == [
        (3, 7, "function", "curly"),
        (14, 16, "method", "Base::after"),
        (18, 25, "function", "make"),
        (20, 22, "method", "class@anonymous::inner"),
        (27, 30, "method", "Real::m"),
        (34, 36, "function", "global_fn"),
    ]
    assert skips == [] and scanned == 1
    u = next(u for u in units if u["name"] == "Real::m")
    assert u["id"] == "edge.php:27:Real::m" and u["loc"] == 4 and not u["truncated"]
    assert u["code"].startswith("function m(") and "function ( $y )" in u["code"]


def _w(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


FN = "<?php\n/** Doc. */\nfunction big() {\n%s\treturn 1;\n}\n"


def test_skips_templates_truncation(tmp_path):
    for rel in ("vendor/a.php", "node_modules/b/c.php", "tests/x.php", "test-x.php", "FooTest.php",
                "build/index.asset.php", "dist/y.php", "x.min.php", "abtest.php"):
        _w(tmp_path, rel, FN % "")
    _w(tmp_path, "t.php", "<?php\nclass T extends WP_UnitTestCase {}\n")
    _w(tmp_path, "m.php", "<?php " + "$a=1;" * 600)
    _w(tmp_path, "tiny.php", "<?php\nfunction t() { return 1; }\n")
    _w(tmp_path, "tpl.php", "<?php\n// comment\n?>\n<div><?php esc_html_e( 'Hi', 'x' ); ?></div>\n")
    _w(tmp_path, "huge.php", FN % ("\t$x = 1;\n" * 3000))
    _w(tmp_path, "bigtpl.php", "<?php\n" + "echo 1;\n" * 3000)
    os.symlink(tmp_path / "tiny.php", tmp_path / "link.php")
    units, skips, scanned = collect_units(tmp_path)
    by = {(s["file"], s["reason"]) for s in skips}
    assert by == {("vendor/a.php", "vendor"), ("node_modules/b/c.php", "node_modules"), ("tests/x.php", "test"),
                  ("test-x.php", "test"), ("FooTest.php", "test"), ("build/index.asset.php", "build"),
                  ("dist/y.php", "build"), ("x.min.php", "minified"), ("t.php", "test"), ("m.php", "minified"),
                  ("tiny.php", "tiny"), ("bigtpl.php", "too_large")}
    assert all(s["line"] == 0 and s["name"] == "" for s in skips if s["reason"] not in ("tiny", "too_large"))
    assert scanned == 15  # symlink not walked
    u = {x["file"]: x for x in units}
    assert set(u) == {"abtest.php", "tpl.php", "huge.php"}  # abtest.php: r1 false-positive fix
    assert u["tpl.php"]["kind"] == "template" and u["tpl.php"]["name"] == "<template>"
    h = u["huge.php"]
    assert h["truncated"] and h["code"].startswith("/** Doc. */") and h["code"].endswith("[truncated by wpjudge]\n")
    assert len(h["code"]) <= 15000 + 40 and h["loc"] == 3003  # lines 3..3005
    units2, _, _ = collect_units(tmp_path, load(templates=False))
    assert "tpl.php" not in {x["file"] for x in units2}


def test_load_dir_and_good_archives(tmp_path):
    src = tmp_path / "my-theme"
    _w(src, "style.css", "/*\nTheme Name: X\n*/")
    _w(src, "functions.php", FN % "")
    root, meta = load_target(src)
    assert root == src and meta == {"name": "my-theme", "type": "theme", "sha256": ""}
    z = tmp_path / "p.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("my-plugin/a.php", FN % "")
    root, meta = load_target(z, tmp_path / "job1")
    assert root == tmp_path / "job1" / "src" / "my-plugin" and meta["name"] == "my-plugin"
    assert meta["type"] == "plugin" and len(meta["sha256"]) == 64
    assert len(collect_units(root)[0]) == 1
    t = tmp_path / "p.tar.gz"
    with tarfile.open(t, "w:gz") as f:
        f.add(src, arcname="my-theme")
    root, meta = load_target(t, tmp_path / "job2")
    assert meta["name"] == "my-theme" and meta["type"] == "theme" and (root / "functions.php").is_file()


def _zip(path, entries, mode=None):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as f:
        for name, data in entries:
            zi = zipfile.ZipInfo(name)
            if mode:
                zi.external_attr = mode << 16
            f.writestr(zi, data, zipfile.ZIP_DEFLATED)


def _tar(path, member, data=b""):
    with tarfile.open(path, "w:gz") as f:
        f.addfile(member, io.BytesIO(data) if data else None)


def _bad_archives(d):
    _zip(d / "slip.zip", [("../../evil.php", "x")])
    _zip(d / "abs.zip", [("/tmp/evil.php", "x")])
    _zip(d / "link.zip", [("a/l.php", "/etc/passwd")], mode=0o120777)
    _zip(d / "bomb.zip", [("a/z.php", b"\0" * 30 * 2**20)])  # 30 MB at ~1000x
    _zip(d / "many.zip", [(f"a/{i}.php", "") for i in range(30)])
    ti = tarfile.TarInfo("../evil.php"); ti.size = 1
    _tar(d / "slip.tar.gz", ti, b"x")
    ti = tarfile.TarInfo("a/l.php"); ti.type = tarfile.SYMTYPE; ti.linkname = "/etc/passwd"
    _tar(d / "sym.tar.gz", ti)
    ti = tarfile.TarInfo("a/h.php"); ti.type = tarfile.LNKTYPE; ti.linkname = "/etc/passwd"
    _tar(d / "hard.tar.gz", ti)
    (d / "notes.txt").write_text("not an archive")
    (d / "fake.zip").write_bytes(b"PK\x03\x04garbage")


def test_malicious_archives_rejected(tmp_path):
    arc = tmp_path / "arc"
    arc.mkdir()
    _bad_archives(arc)
    cfg = load(max_files=20)
    for a in sorted(arc.iterdir()):
        dest = tmp_path / "jobs" / a.name
        with pytest.raises(Rejected):
            load_target(a, dest, cfg)
    with pytest.raises(Rejected):
        load_target(tmp_path / "missing.zip", tmp_path / "jobs" / "m")
    outside = {p for p in tmp_path.rglob("*") if "jobs" not in p.relative_to(tmp_path).parts}
    assert outside == {arc, *arc.iterdir()}  # nothing escaped
    assert not list((tmp_path / "jobs").rglob("*.php"))  # nothing extracted (guards run before extractall)
    assert not Path("/tmp/evil.php").exists()


def test_oversize_total(tmp_path):
    z = tmp_path / "big.zip"
    _zip(z, [("a/x.php", os.urandom(2 * 2**20))])
    with pytest.raises(Rejected):
        load_target(z, tmp_path / "j", load(max_total_mb=1))


# ---- fix round 1 regressions (EVAL_W1.md) ----

def test_docblock_drops_attributes_and_modifiers(tmp_path):
    _w(tmp_path, "c.php", "<?php\nclass C {\n\t/**\n\t * Doc.\n\t */\n\t#[Pure]\n\tfinal public static function s() {\n"
                          "\t\treturn 1;\n\t}\n\tprivate function p() {\n\t\treturn 2;\n\t}\n}\n")
    units, _, _ = collect_units(tmp_path)
    s, p = sorted(units, key=lambda u: u["line"])
    assert s["name"] == "C::s" and s["code"].startswith("/**") and "*/\nfunction s()" in s["code"]
    assert "public" not in s["code"] and "#[Pure]" not in s["code"]
    assert p["code"].startswith("function p()")


def test_tar_count_guard_streams(tmp_path, monkeypatch):
    # guard must trip while streaming headers, not after getmembers() has buffered the whole archive
    t = tmp_path / "many.tar.gz"
    with tarfile.open(t, "w:gz") as f:
        for i in range(50):
            f.addfile(tarfile.TarInfo(f"a/{i}.php"))
    seen = []
    real_next = tarfile.TarFile.next
    monkeypatch.setattr(tarfile.TarFile, "next", lambda self: seen.append(1) or real_next(self))
    monkeypatch.setattr(tarfile.TarFile, "getmembers", lambda self: pytest.fail("getmembers before guard"))
    with pytest.raises(Rejected, match="too many files"):
        load_target(t, tmp_path / "j", load(max_files=20))
    assert len(seen) < 25  # stopped near member 21 of 50 (plus open/is_tarfile probes)


def test_macosx_zip_unwrapped(tmp_path):
    z = tmp_path / "mac.zip"
    _zip(z, [("slug/style.css", "/*\nTheme Name: S\n*/"), ("slug/functions.php", FN % ""),
             ("__MACOSX/slug/._functions.php", "\0\5junk")])
    root, meta = load_target(z, tmp_path / "j")
    assert meta["name"] == "slug" and meta["type"] == "theme"
    units, skips, scanned = collect_units(root)
    assert scanned == 1 and skips == [] and units[0]["file"] == "functions.php"


def test_nested_owner_not_template_and_orphan_error_and_upper_php(tmp_path):
    _w(tmp_path, "ns.php", "<?php\nnamespace Foo {\ninterface Bar {\n\tpublic function a();\n}\n}\n")
    _w(tmp_path, "cond.php", "<?php\nif ( ! class_exists( 'X' ) ) {\n\tclass X {\n\t}\n}\n")
    _w(tmp_path, "err.php", "<?php\nfunction bad( {\n\treturn 1;\n\nfunction ok() {\n\treturn 1;\n}\n")
    _w(tmp_path, "UP.PHP", FN % "")
    units, skips, scanned = collect_units(tmp_path)
    assert scanned == 4
    assert {u["file"] for u in units} >= {"UP.PHP"} and not {"ns.php", "cond.php"} & {u["file"] for u in units}
    assert not [s for s in skips if s["file"] in ("ns.php", "cond.php")]
    assert {"file": "err.php", "line": 0, "name": "", "reason": "parse_error"} in skips


def test_rejected_without_dest_leaves_no_tempdir(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _zip(tmp_path / "slip.zip", [("../../evil.php", "x")])
    with pytest.raises(Rejected):
        load_target(tmp_path / "slip.zip")
    assert not list(tmp_path.glob("wpjudge-*"))
