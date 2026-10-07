"""python scripts/verify_assets.py [--root .] [--kit demo|full] -- check the downloaded kit files.

Offline, read-only and standard-library only: it makes no network call, writes nothing and imports
neither torch, ultralytics, mediapipe nor cv2. For every required file it prints OK, MISSING or
BAD HASH with the exact folder to put the file in and the Kaggle dataset it comes from, then a
final line READY or NOT READY (exit code 0 / 1).

    python scripts/verify_assets.py                 the demo kit (weights, clips, playlist)
    python scripts/verify_assets.py --kit full      the demo kit plus run videos and dataset

The expected hashes of the demo kit are the committed file assets/ASSETS.sha256 (one line per file:
``<sha256>  <repo-relative path>``, ``#`` comments; a header line ``# dataset: <slug>`` names the
Kaggle dataset). The weights are also checked against weights/MANIFEST.json. It complements
``python scripts/demo.py --check`` (which loads the models) and never replaces it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ASSETS_LIST = Path("assets") / "ASSETS.sha256"
DEMO_SLUG_DEFAULT = "<kaggle-user>/bas-demo-kit"
FULL_SLUG_DEFAULT = "<kaggle-user>/bas-dataset-full"

_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")
_SEARCH_DIRS = (".", "weights", "demo_videos", "kit_staging/bas-demo-kit")

# What the full kit needs besides the demo kit and the run videos (existence only, no hash list).
FULL_EXISTS = (
    "runs/manifest.csv",
    "data/dataset",
    "data/corrections",
    "data/label_review.csv",
    "reports/dataset.json",
)


@dataclass(frozen=True)
class Entry:
    rel: str  # repo-relative, always with forward slashes
    sha256: str
    dataset: str = ""


@dataclass(frozen=True)
class Result:
    status: str  # OK | MISSING | BAD HASH
    rel: str
    detail: str = ""


def sha256_of_file(path: Path) -> str:
    """The same streamed sha256 as ``contracts.sha256_of_file`` (kept here so this script needs
    nothing but the standard library)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_rel(raw: str) -> str:
    """A list path in Windows or POSIX form -> forward slashes, no leading ``./``."""
    return PurePosixPath(raw.strip().replace("\\", "/")).as_posix().removeprefix("./")


def parse_assets(text: str) -> tuple[str, str, list[Entry]]:
    """(demo dataset slug, full dataset slug, entries) from the text of assets/ASSETS.sha256.
    Raises ValueError (with the line number) on a line that is neither a comment nor a hash line."""
    slug, full = DEMO_SLUG_DEFAULT, FULL_SLUG_DEFAULT
    entries: list[Entry] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.match(r"#\s*dataset(-full)?:\s*(\S+)", line)
            if m:
                if m.group(1):
                    full = m.group(2)
                else:
                    slug = m.group(2)
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"line {number}: not '<sha256>  <path>': {line[:60]!r}")
        entries.append(Entry(rel=norm_rel(m.group(2)), sha256=m.group(1).lower()))
    return slug, full, entries


def _folder(rel: str) -> str:
    parent = PurePosixPath(rel).parent.as_posix()
    return "the repository root" if parent == "." else parent + "/"


def _where_else(root: Path, rel: str) -> str:
    name = PurePosixPath(rel).name
    for d in _SEARCH_DIRS:
        cand = root / d / name
        if cand.is_file() and cand != root / rel:
            return PurePosixPath(cand.relative_to(root).as_posix()).as_posix()
    return ""


def check_file(root: Path, entry: Entry, dataset: str) -> Result:
    path = root / entry.rel
    where = f"put it in {_folder(entry.rel)} (from the Kaggle dataset {dataset})"
    if not path.is_file():
        other = _where_else(root, entry.rel)
        hint = f"found at {other}: move it to {_folder(entry.rel)}" if other else where
        return Result("MISSING", entry.rel, hint)
    if path.stat().st_size == 0:
        return Result("BAD HASH", entry.rel, f"empty file (0 bytes): download it again; {where}")
    got = sha256_of_file(path)
    if got != entry.sha256:
        return Result(
            "BAD HASH",
            entry.rel,
            f"sha256 {got[:12]}.. expected {entry.sha256[:12]}..: partial or wrong file, "
            f"download it again; {where}",
        )
    return Result("OK", entry.rel)


def _manifest_hashes(root: Path) -> dict[str, str]:
    """file name -> sha256 for every weights file named in weights/MANIFEST.json."""
    data = json.loads((root / "weights" / "MANIFEST.json").read_text(encoding="utf-8"))
    out = {d["file"]: d["sha256"] for d in data.get("detectors", [])}
    if "hand" in data:
        out[data["hand"]["file"]] = data["hand"]["sha256"]
    return out


def _run_entries(root: Path) -> list[Entry]:
    with (root / "runs" / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        return [
            Entry(rel=f"runs/{r['run_id']}/video.mp4", sha256=r["video_sha256"].lower())
            for r in csv.DictReader(fh)
        ]


def verify(root: Path, kit: str) -> tuple[list[Result], list[str]]:
    """All results plus the problems that make the check meaningless (those force NOT READY)."""
    problems: list[str] = []
    results: list[Result] = []
    listing = root / ASSETS_LIST
    if not listing.is_file():
        return [], [f"{ASSETS_LIST.as_posix()} not found under {root}: is --root the repository?"]
    try:
        demo_slug, full_slug, entries = parse_assets(listing.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        return [], [f"{ASSETS_LIST.as_posix()}: {exc}"]
    if not entries:
        return [], [f"{ASSETS_LIST.as_posix()} lists no files"]
    try:
        manifest = _manifest_hashes(root)
    except (OSError, ValueError, KeyError) as exc:
        manifest = {}
        problems.append(
            f"weights/MANIFEST.json cannot be read ({type(exc).__name__}): git pull again"
        )
    for entry in entries:
        name = PurePosixPath(entry.rel).name
        if name in manifest and manifest[name] != entry.sha256:
            results.append(
                Result(
                    "BAD HASH",
                    entry.rel,
                    "assets/ASSETS.sha256 disagrees with weights/MANIFEST.json "
                    f"for {name}: update the repository (git pull)",
                )
            )
            continue
        results.append(check_file(root, entry, demo_slug))
    if kit == "full":
        results.extend(_verify_full(root, full_slug, problems))
    return results, problems


def _verify_full(root: Path, dataset: str, problems: list[str]) -> list[Result]:
    results: list[Result] = []
    if not (root / "runs" / "manifest.csv").is_file():
        problems.append("runs/manifest.csv not found: git pull again")
    else:
        for entry in _run_entries(root):
            results.append(check_file(root, entry, dataset))
    for rel in FULL_EXISTS:
        path = root / rel
        ok = path.is_file() or (path.is_dir() and any(path.iterdir()))
        results.append(
            Result("OK", rel)
            if ok
            else Result(
                "MISSING", rel, f"put it in {_folder(rel)} (from the Kaggle dataset {dataset})"
            )
        )
    return results


def render(results: list[Result], problems: list[str]) -> tuple[list[str], bool]:
    lines = [
        f"[{r.status}]".ljust(11) + r.rel + (f"  -> {r.detail}" if r.detail else "")
        for r in results
    ]
    lines += [f"[PROBLEM]  {p}" for p in problems]
    ready = bool(results) and not problems and all(r.status == "OK" for r in results)
    n_ok = sum(r.status == "OK" for r in results)
    lines.append(f"{n_ok}/{len(results)} files OK")
    lines.append("READY" if ready else "NOT READY")
    return lines, ready


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline check of the kit files against assets/ASSETS.sha256 and MANIFEST.json."
    )
    ap.add_argument(
        "--root", default=str(ROOT), help="the repository folder (default: this repository)"
    )
    ap.add_argument(
        "--kit",
        choices=["demo", "full"],
        default="demo",
        help="demo: weights, clips, playlist (default); full: also the run videos and the dataset",
    )
    args = ap.parse_args(argv)
    results, problems = verify(Path(args.root), args.kit)
    lines, ready = render(results, problems)
    print("\n".join(lines))
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
