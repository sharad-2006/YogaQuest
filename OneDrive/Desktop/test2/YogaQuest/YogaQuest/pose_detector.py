"""
pose_detector.py — YOLO pose detection via Ultralytics (Python 3.13 compatible).

Converts COCO-17 keypoints into the MediaPipe-compatible 33-point layout used
by poses_config.py so all angle checks and scoring remain unchanged.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

# COCO 17 keypoint index -> MediaPipe 33-point index (used by poses_config)
COCO_TO_MP: dict[int, int] = {
    0: 0,    # nose
    1: 2,    # left eye
    2: 5,    # right eye
    3: 7,    # left ear
    4: 8,    # right ear
    5: 11,   # left shoulder
    6: 12,   # right shoulder
    7: 13,   # left elbow
    8: 14,   # right elbow
    9: 15,   # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
    15: 27,  # left ankle
    16: 28,  # right ankle
}

# Skeleton connections for overlay drawing (COCO indices)
COCO_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MODEL = os.path.join(_MODULE_DIR, "yolo11n-pose.pt")
if not os.path.exists(_LOCAL_MODEL):
    _LOCAL_MODEL = os.path.join(_MODULE_DIR, "yolo11s-pose.pt")

DEFAULT_MODEL = os.environ.get(
    "YOGA_POSE_MODEL",
    _LOCAL_MODEL if os.path.exists(_LOCAL_MODEL) else "yolo11s-pose.pt",
)
DEFAULT_IMGSZ = int(os.environ.get("YOGA_POSE_IMGSZ", "416"))


class PoseDetectionResult:
    """Container returned by YoloPoseModel.process()."""

    __slots__ = ("landmarks", "keypoints_px", "keypoints_conf", "image_shape")

    def __init__(
        self,
        landmarks: Optional[np.ndarray],
        keypoints_px: Optional[np.ndarray],
        keypoints_conf: Optional[np.ndarray],
        image_shape: Tuple[int, int],
    ):
        self.landmarks = landmarks          # (33, 4) normalised MP layout
        self.keypoints_px = keypoints_px    # (17, 2) pixel coords
        self.keypoints_conf = keypoints_conf  # (17,)
        self.image_shape = image_shape      # (height, width)


class YoloPoseModel:
    """
    Drop-in replacement for MediaPipe Pose.

    Parameters mirror the old MediaPipe API where relevant:
      static_image_mode  — disable tracking for single-frame capture
      min_detection_confidence — YOLO confidence threshold
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,          # kept for API compat, ignored
        smooth_landmarks: bool = True,      # kept for API compat, ignored
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,  # kept for API compat, ignored
        enable_segmentation: bool = False,  # kept for API compat, ignored
        model_name: Optional[str] = None,
    ):
        from ultralytics import YOLO

        self.conf = min_detection_confidence
        self.static = static_image_mode
        self.model = YOLO(model_name or DEFAULT_MODEL)

    def process(self, frame_bgr: np.ndarray) -> PoseDetectionResult:
        """Run pose detection on a BGR frame."""
        h, w = frame_bgr.shape[:2]
        kwargs = dict(conf=self.conf, verbose=False, imgsz=DEFAULT_IMGSZ)

        if self.static:
            preds = self.model(frame_bgr, **kwargs)
        else:
            preds = self.model.track(frame_bgr, persist=True, **kwargs)

        return self._parse(preds, w, h)

    def _parse(self, preds, width: int, height: int) -> PoseDetectionResult:
        empty = PoseDetectionResult(None, None, None, (height, width))

        if not preds or len(preds) == 0:
            return empty

        kpts = preds[0].keypoints
        if kpts is None or kpts.data is None or len(kpts.data) == 0:
            return empty

        data = kpts.data.cpu().numpy()  # (N, 17, 3)  x, y, conf

        # Pick the person with the highest mean keypoint confidence
        if len(data) > 1:
            idx = int(np.argmax(data[:, :, 2].mean(axis=1)))
        else:
            idx = 0

        person = data[idx]
        kpx, kpy, kpc = person[:, 0], person[:, 1], person[:, 2]

        if float(kpc.mean()) < 0.25:
            return empty

        keypoints_px = np.stack([kpx, kpy], axis=-1)

        landmarks = np.zeros((33, 4), dtype=float)
        for coco_i, mp_i in COCO_TO_MP.items():
            if kpc[coco_i] > 0:
                landmarks[mp_i] = [
                    kpx[coco_i] / width,
                    kpy[coco_i] / height,
                    0.0,
                    float(kpc[coco_i]),
                ]

        return PoseDetectionResult(landmarks, keypoints_px, kpc, (height, width))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


def draw_skeleton(
    frame: np.ndarray,
    result: PoseDetectionResult,
    color: Tuple[int, int, int],
    thickness: int = 2,
    min_conf: float = 0.30,
) -> None:
    """Draw COCO skeleton on *frame* in-place."""
    if result.keypoints_px is None or result.keypoints_conf is None:
        return

    kpx = result.keypoints_px
    conf = result.keypoints_conf

    for i, j in COCO_CONNECTIONS:
        if conf[i] >= min_conf and conf[j] >= min_conf:
            pt1 = (int(kpx[i][0]), int(kpx[i][1]))
            pt2 = (int(kpx[j][0]), int(kpx[j][1]))
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    for i in range(len(conf)):
        if conf[i] >= min_conf:
            cv2.circle(
                frame,
                (int(kpx[i][0]), int(kpx[i][1])),
                3,
                color,
                -1,
                cv2.LINE_AA,
            )
