"""Thin Grounding DINO tiny wrapper for the P1.2 spike (ISSUES.md 2026-09-28
"licenses of auto-labeling tools not yet verified"; license recorded there
by this spike). Library calls only -- no re-implementation, no tests here
(the model is real and this is not library-pure code); see
training/spikes/postprocess.py for the tested helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from training.spikes.postprocess import RawDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


@dataclass
class DetectResult:
    detections: list[RawDetection]
    seconds: float


class GroundingDinoSpike:
    """Loads once; ``detect`` runs one forward pass per call. Callers scope
    each call to a single phrase or a single class's own phrasings (see
    training/spikes/postprocess.py's ``select_best_in_band`` docstring for
    why: Grounding DINO's returned text span does not reliably echo the
    submitted phrase, so a single multi-class query's outputs cannot be
    reliably attributed back to a class by text alone)."""

    def __init__(self, box_threshold: float = 0.25, text_threshold: float = 0.20) -> None:
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        # The processor's default resize (shortest_edge=800, longest_edge=1333)
        # upscales this rig's 848x480 frames to 1333x755 and takes ~12s/call
        # on this CPU -- far too slow for a 600-call time-boxed spike.
        # shortest_edge=480 (close to the frame's native height, so this is
        # a mild resize rather than an upscale) measured at ~3.4s/call on a
        # real frame with the same detection surviving at a slightly higher
        # score (0.588 vs 0.570) -- checkpoint 2's report should re-check
        # this if the smallest object (the ~6cm START card) needs more
        # resolution than this affords.
        self.processor.image_processor.size = {"shortest_edge": 480, "longest_edge": 800}
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
        self.model.eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, image_rgb: np.ndarray, phrases: list[str]) -> DetectResult:
        """``phrases`` are already lowercase and each ends with a period
        (Grounding DINO's documented input convention)."""
        text = " ".join(phrases)
        height, width = image_rgb.shape[:2]
        start = time.perf_counter()
        inputs = self.processor(images=image_rgb, text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # transformers 5.17.0 renamed the model card's documented
        # `box_threshold` kwarg to `threshold` (verified via
        # inspect.signature against the installed version).
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]
        seconds = time.perf_counter() - start

        boxes = results["boxes"].tolist()
        scores = results["scores"].tolist()
        labels = results["labels"]
        detections = [
            RawDetection(phrase=label.strip(), score=float(score), box=tuple(box))
            for label, score, box in zip(labels, scores, boxes, strict=True)
        ]
        return DetectResult(detections=detections, seconds=seconds)
