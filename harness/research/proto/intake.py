"""Prototype archive intake guard (stdlib only)."""
import tarfile, zipfile
from pathlib import Path, PurePosixPath

MAX_TOTAL = 200 * 2**20   # uncompressed bytes
MAX_RATIO = 100           # uncompressed / compressed
MAX_FILES = 20000


class Rejected(ValueError):
    pass


def _check_name(name: str):
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
        raise Rejected(f"unsafe path: {name}")


def _check_totals(total, n, packed):
    if n > MAX_FILES:
        raise Rejected("too many files")
    if total > MAX_TOTAL or (total > 10 * 2**20 and total > MAX_RATIO * packed):
        raise Rejected("uncompressed size/ratio limit")


def extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=False)
    packed = archive.stat().st_size
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            for i in infos:
                _check_name(i.filename)
                if (i.external_attr >> 16) & 0o170000 == 0o120000:
                    raise Rejected(f"symlink: {i.filename}")
            _check_totals(sum(i.file_size for i in infos), len(infos), packed)
            z.extractall(dest)  # declared sizes checked above; zipfile enforces them on read (CRC/size)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:  # handles .tar, .tar.gz, .tgz
            members = t.getmembers()
            for m in members:
                _check_name(m.name)
                if m.issym() or m.islnk():
                    raise Rejected(f"link: {m.name}")
            _check_totals(sum(m.size for m in members), len(members), packed)
            t.extractall(dest, filter="data")  # second line of defence
    else:
        raise Rejected("not a zip or tar archive")
    # unwrap the usual single top-level folder (plugin-slug/)
    kids = [p for p in dest.iterdir()]
    return kids[0] if len(kids) == 1 and kids[0].is_dir() else dest
