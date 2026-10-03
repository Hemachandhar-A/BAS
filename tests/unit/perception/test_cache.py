"""F14 stage 9 / F2 (essential-features.md): perception/cache.py -- the cache writer, the reader
that refuses a mismatching header, the atomic and resumable run builder. Fakes only (no video,
no model)."""

from __future__ import annotations

import csv
import json
import logging

import numpy as np
import pytest

from contracts import Detection, Frame, PerceptionCacheHeader, PerceptionFrame, SourceError
from perception import cache as C

pytestmark = [pytest.mark.F2, pytest.mark.F14]

STAMP = "yolo11n:aaaaaaaa|hand:bbbbbbbb|pose:none"
OTHER = "rfdetr-nano:cccccccc|hand:bbbbbbbb|pose:none"


def pframe(i: int) -> PerceptionFrame:
    return PerceptionFrame(
        frame_id=i,
        t=i / 30,
        detections=[Detection(label="tray", conf=0.5, box=(1.0, 2.0, 3.0 + i, 4.0))],
    )


def header(**kw) -> PerceptionCacheHeader:
    d = {"run_id": "x001", "experiment_id": "exp", "fps": 10.0, "model_stamp": STAMP}
    return PerceptionCacheHeader(**{**d, **kw})


# --- writer and reader ----------------------------------------------------------------------


def test_round_trip_header_then_frames(tmp_path):
    p = C.cache_path(tmp_path, "x001")
    assert p == tmp_path / "x001" / "perception.jsonl"
    frames = [pframe(i) for i in (0, 3, 6)]
    assert C.write_cache(p, header(), frames) == 3
    h, got = C.read_cache(p)
    assert h == header() and got == frames
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["model_stamp"] == STAMP


def test_empty_frame_list_is_a_valid_cache(tmp_path):
    p = C.cache_path(tmp_path, "x001")
    C.write_cache(p, header(), [])
    assert C.read_cache(p)[1] == []


def test_the_same_input_writes_identical_bytes_with_unix_newlines(tmp_path):
    a, b = C.cache_path(tmp_path / "a", "r"), C.cache_path(tmp_path / "b", "r")
    for p in (a, b):
        C.write_cache(p, header(), [pframe(i) for i in range(4)])
    assert a.read_bytes() == b.read_bytes() and b"\r" not in a.read_bytes()


def test_reader_accepts_a_matching_stamp_and_fps_and_refuses_others(tmp_path):
    p = C.cache_path(tmp_path, "x001")
    C.write_cache(p, header(), [pframe(0)])
    C.read_cache(p, expected_stamp=STAMP, expected_fps=10.0)
    with pytest.raises(C.CacheMismatch, match="model_stamp") as e:
        C.read_cache(p, expected_stamp=OTHER)
    assert STAMP in str(e.value) and OTHER in str(e.value)
    with pytest.raises(C.CacheMismatch, match="fps"):
        C.read_cache(p, expected_fps=15.0)


def test_header_alone_can_be_read_and_checked_without_the_frames(tmp_path):
    p = C.cache_path(tmp_path, "x001")
    C.write_cache(p, header(), [pframe(0)])
    assert C.read_header(p, expected_stamp=STAMP) == header()
    with pytest.raises(C.CacheMismatch):
        C.read_header(p, expected_stamp=OTHER)


@pytest.mark.parametrize(
    "content,match",
    [
        ("", "empty"),
        ("not json\n", "header"),
        ('{"run_id": "x"}\n', "header"),
        (header().model_dump_json() + "\n{broken\n", "line 2"),
        (header().model_dump_json() + '\n{"frame_id": -1, "t": 0.0}\n', "line 2"),
    ],
)
def test_a_malformed_file_gives_a_clear_error(tmp_path, content, match):
    p = tmp_path / "perception.jsonl"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(C.CacheError, match=match):
        C.read_cache(p)


def test_a_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(C.CacheError, match="not found"):
        C.read_cache(tmp_path / "nope.jsonl")


def test_frame_ids_must_increase(tmp_path):
    p = tmp_path / "perception.jsonl"
    p.write_text(
        header().model_dump_json() + "\n" + pframe(3).model_dump_json() + "\n"
        + pframe(3).model_dump_json() + "\n",
        encoding="utf-8",
    )  # fmt: skip
    with pytest.raises(C.CacheError, match="increas"):
        C.read_cache(p)


# --- atomic writes --------------------------------------------------------------------------


def test_a_failure_while_writing_leaves_no_file_and_keeps_the_old_one(tmp_path):
    p = C.cache_path(tmp_path, "x001")
    C.write_cache(p, header(), [pframe(0)])
    old = p.read_bytes()

    def broken():
        yield pframe(0)
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        C.write_cache(p, header(fps=5.0), broken())
    assert p.read_bytes() == old
    assert [f.name for f in p.parent.iterdir()] == ["perception.jsonl"]  # no temp file left
    fresh = C.cache_path(tmp_path, "x002")
    with pytest.raises(RuntimeError):
        C.write_cache(fresh, header(run_id="x002"), broken())
    assert not fresh.exists() and not any(fresh.parent.iterdir())


def test_is_complete_needs_a_matching_header_and_an_intact_last_line(tmp_path):
    p = C.cache_path(tmp_path, "x001")
    assert not C.is_complete(p, STAMP, 10.0)
    C.write_cache(p, header(), [pframe(0), pframe(3)])
    assert C.is_complete(p, STAMP, 10.0)
    assert not C.is_complete(p, OTHER, 10.0) and not C.is_complete(p, STAMP, 15.0)
    p.write_bytes(p.read_bytes()[:-20])  # cut in the middle of the last frame
    assert not C.is_complete(p, STAMP, 10.0)


# --- building from a source -----------------------------------------------------------------


class FakeSource:
    def __init__(self, n: int, fps: float | None = 30.0):
        self.fps, self.n, self.i, self.exhausted, self.closed = fps, n, 0, False, False

    def read(self):
        if self.i >= self.n:
            self.exhausted = True
            return None
        f = Frame(
            frame_id=self.i,
            t=self.i / (self.fps or 30.0),
            image=np.full((4, 4, 3), self.i, np.uint8),
        )
        self.i += 1
        return f

    def close(self):
        self.closed = True


class FakePipeline:
    model_stamp = STAMP

    def __init__(self, fail_on=()):
        self.fail_on, self.resets, self.seen = set(fail_on), 0, []

    def process(self, frame):
        self.seen.append(frame.frame_id)
        if frame.frame_id in self.fail_on:
            raise RuntimeError("bad frame")
        return pframe(frame.frame_id)

    def reset(self):
        self.resets += 1


def build(tmp_path, pipe, n=31, fps=10.0, src_fps=30.0, **kw):
    return C.build_run(
        "x001", "video.mp4", pipe, fps=fps, experiment_id="exp", out_root=tmp_path,
        source_factory=lambda _: FakeSource(n, src_fps), **kw,
    )  # fmt: skip


def test_build_decimates_to_the_target_fps_and_records_it_in_the_header(tmp_path):
    pipe = FakePipeline()
    r = build(tmp_path, pipe)
    assert pipe.seen == list(range(0, 31, 3)) and pipe.resets == 1
    assert r["status"] == "built" and r["frames"] == 11 and r["every"] == 3
    h, frames = C.read_cache(
        C.cache_path(tmp_path, "x001"), expected_stamp=STAMP, expected_fps=10.0
    )
    assert (h.run_id, h.experiment_id, h.fps) == ("x001", "exp", 10.0)
    assert [f.frame_id for f in frames] == list(range(0, 31, 3))


def test_an_existing_complete_cache_is_skipped_unless_forced(tmp_path):
    build(tmp_path, FakePipeline())
    again = FakePipeline()
    assert build(tmp_path, again)["status"] == "skipped" and again.seen == []
    forced = FakePipeline()
    assert build(tmp_path, forced, force=True)["status"] == "built" and forced.seen


def test_a_cache_with_another_stamp_or_fps_is_rebuilt(tmp_path):
    build(tmp_path, FakePipeline())
    p = FakePipeline()
    p.model_stamp = OTHER
    assert build(tmp_path, p)["status"] == "built"
    assert C.read_header(C.cache_path(tmp_path, "x001")).model_stamp == OTHER
    assert build(tmp_path, p, fps=15.0)["status"] == "built"


def test_a_failing_frame_is_skipped_and_counted_not_fatal(tmp_path, caplog):
    pipe = FakePipeline(fail_on={3})
    with caplog.at_level(logging.WARNING, logger="perception.cache"):
        r = build(tmp_path, pipe)
    assert r["frames"] == 10 and r["skipped_frames"] == 1
    assert [f.frame_id for f in C.read_cache(C.cache_path(tmp_path, "x001"))[1]][:2] == [0, 6]
    assert any("frame 3" in m.getMessage() for m in caplog.records)


def test_the_source_is_closed_even_when_the_build_fails(tmp_path):
    src = FakeSource(10)

    class Dying(FakePipeline):
        def process(self, frame):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        C.build_run("x001", "v", Dying(), fps=10.0, experiment_id="e", out_root=tmp_path,
                    source_factory=lambda _: src)  # fmt: skip
    assert src.closed and not C.cache_path(tmp_path, "x001").exists()


def test_an_unusable_target_fps_or_missing_source_fps(tmp_path):
    with pytest.raises(ValueError):
        build(tmp_path, FakePipeline(), fps=0.0)
    r = build(tmp_path, FakePipeline(), src_fps=None, fps=10.0)  # falls back to the capture rate
    assert r["every"] == 3


def test_target_above_source_rate_keeps_every_frame(tmp_path):
    assert build(tmp_path, FakePipeline(), fps=60.0)["every"] == 1


# --- choosing runs --------------------------------------------------------------------------


def _manifest(tmp_path):
    p = tmp_path / "manifest.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "split", "fps"])
        for rid, sp in [("x001", "train"), ("x002", "val"), ("x003", "test"), ("x004", "val")]:
            w.writerow([rid, sp, 30])
    return p


def test_select_runs_by_split_run_or_all(tmp_path):
    m = _manifest(tmp_path)
    ids = lambda **kw: [r for r, _ in C.select_runs(m, tmp_path, **kw)]  # noqa: E731
    assert ids(split="val") == ["x002", "x004"] and ids(split="all") == [
        "x001",
        "x002",
        "x003",
        "x004",
    ]
    assert ids(split="all", run="x003") == ["x003"]
    assert C.select_runs(m, tmp_path, split="train")[0][1] == tmp_path / "x001" / "video.mp4"
    with pytest.raises(ValueError, match="x999"):
        ids(split="all", run="x999")
    with pytest.raises(ValueError, match="split"):
        ids(split="nope")


def test_build_runs_continues_after_a_failed_run_and_reports_it(tmp_path):
    def factory(path):
        if "x002" in str(path):
            raise SourceError("cannot open")
        return FakeSource(7)

    results = C.build_runs(
        [("x001", tmp_path / "x001" / "video.mp4"), ("x002", tmp_path / "x002" / "video.mp4")],
        FakePipeline(), fps=10.0, experiment_id="e", out_root=tmp_path, source_factory=factory,
    )  # fmt: skip
    assert [r["status"] for r in results] == ["built", "failed"]
    assert "cannot open" in results[1]["error"]
