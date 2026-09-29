"""harness/fakes.py -- FakePerception and FakeFrameSource, both duck-typed
implementations of the contracts.py Perception/FrameSource Protocols."""

from __future__ import annotations

from contracts import FrameSource, Perception, PerceptionFrame
from harness.fakes import FakeFrameSource, FakePerception, make_blank_frame


def test_fake_perception_satisfies_the_protocol() -> None:
    assert isinstance(FakePerception([]), Perception)


def test_fake_frame_source_satisfies_the_protocol() -> None:
    assert isinstance(FakeFrameSource([]), FrameSource)


def test_fake_perception_replays_script_in_order_and_carries_through_frame_id_and_t() -> None:
    script = [
        PerceptionFrame(frame_id=999, t=999.0, detections=[]),
        PerceptionFrame(frame_id=999, t=999.0, detections=[]),
    ]
    fake = FakePerception(script)
    out0 = fake.process(make_blank_frame(0, 0.5))
    out1 = fake.process(make_blank_frame(1, 1.5))
    assert out0.frame_id == 0
    assert out0.t == 0.5
    assert out1.frame_id == 1
    assert out1.t == 1.5


def test_fake_perception_returns_empty_frame_once_script_exhausted() -> None:
    fake = FakePerception([])
    out = fake.process(make_blank_frame(5, 5.0))
    assert out.frame_id == 5
    assert out.detections == []
    assert out.hands == []
    assert out.pose is None


def test_fake_perception_reset_replays_from_the_start() -> None:
    script = [PerceptionFrame(frame_id=0, t=0.0, detections=[])]
    fake = FakePerception(script)
    fake.process(make_blank_frame(0, 0.0))  # consumes the only scripted entry
    fake.reset()
    out = fake.process(make_blank_frame(7, 7.0))
    # After reset the (now only) scripted entry is available again, not the
    # post-exhaustion empty-frame fallback.
    assert out.frame_id == 7


def test_fake_frame_source_yields_items_then_exhausts() -> None:
    f0 = make_blank_frame(0, 0.0)
    source = FakeFrameSource([f0, None, None])
    assert source.read() is f0
    assert source.read() is None
    assert source.exhausted is False  # a transient gap, not end-of-stream
    assert source.read() is None
    assert source.exhausted is False
    assert source.read() is None
    assert source.exhausted is True
    source.close()
    assert source.closed is True
