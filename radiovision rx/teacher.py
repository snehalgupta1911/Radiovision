"""
Teacher network (RF-Pose sense): a camera-based pose estimator whose outputs
become the *labels* the RF student learns to reproduce. No manual annotation.

Currently MediaPipe. You chose Hands originally, but note: recovering hand
pose from 2.4 GHz WiFi CSI is extremely hard (wavelength ~12.5 cm vs a hand a
few cm across). Full-body pose is the realistic RF target, so this defaults to
MediaPipe Pose. Swap back to Hands only if you keep the vision path as a
separate product rather than an RF-supervision signal.
"""
from __future__ import annotations
import numpy as np

try:
    import mediapipe as mp
    import cv2
except ImportError:
    mp = None


class PoseTeacher:
    """Returns (K, 3) array of [x, y, visibility] normalized keypoints, or None."""
    N_KEYPOINTS = 33  # MediaPipe Pose full body

    def __init__(self, detection_confidence: float = 0.5, tracking_confidence: float = 0.5):
        if mp is None:
            raise ImportError("pip install mediapipe opencv-python")
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def __call__(self, frame_bgr) -> np.ndarray | None:
        res = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        return np.array([[p.x, p.y, p.visibility] for p in lm], dtype=np.float32)

    # For an RF-Pose-faithful setup, render these keypoints into gaussian
    # confidence heatmaps and train the student to regress the *maps*, not raw
    # coordinates. Left as a knob because it couples to your model's output head.