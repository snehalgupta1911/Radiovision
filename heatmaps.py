"""
heatmaps.py - camera teacher: MediaPipe Pose -> 14 joints -> 14 heatmaps.

Representation (used identically in collection, training, evaluation, visualization):
  J = 14 joints, RF-Pose style: head, neck, r/l shoulder, r/l elbow, r/l wrist,
      r/l hip, r/l knee, r/l ankle. (neck = midpoint of the shoulders)
  H = W = 32: each heatmap covers the whole camera image (x and y normalized to
      0..1 and scaled to 32 cells, so one cell is ~1/32 of the image width/height).
  Each joint map is a 2-D Gaussian (sigma = 1.5 cells) centred on the joint,
  peak value 1. A joint the teacher can't see (visibility < 0.5 or outside the
  image) gets an all-zero map AND is masked out of the loss, so "not visible"
  is never taught as "not there".
"""
from __future__ import annotations
import os, urllib.request
import numpy as np

JOINTS = ["head", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow",
          "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"]
J = len(JOINTS)
MP_INDEX = [0, None, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27]   # MediaPipe landmark ids
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9),
            (9, 10), (1, 11), (11, 12), (12, 13), (8, 11)]
HM_SIZE = 32
HM_SIGMA = 1.5
VIS_THR = 0.5


def mp33_to_joints(kp33: np.ndarray) -> np.ndarray:
    """(33,4) MediaPipe [x, y, z, visibility] -> (14,3) [x, y, visibility]."""
    out = np.zeros((J, 3), np.float32)
    for i, m in enumerate(MP_INDEX):
        if m is None:
            a, b = kp33[11], kp33[12]
            out[i] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[3], b[3])]
        else:
            out[i] = [kp33[m, 0], kp33[m, 1], kp33[m, 3]]
    return out


def joint_mask(kp: np.ndarray) -> np.ndarray:
    """(...,14,3) -> (...,14) bool: joint confidently seen and inside the image."""
    x, y, v = kp[..., 0], kp[..., 1], kp[..., 2]
    return (v >= VIS_THR) & (x >= -0.05) & (x <= 1.05) & (y >= -0.05) & (y <= 1.05)


def make_heatmaps(kp: np.ndarray, size: int = HM_SIZE, sigma: float = HM_SIGMA) -> np.ndarray:
    """(...,14,3) joints -> (...,14,size,size) float32 Gaussian heatmaps in [0,1]."""
    kp = np.asarray(kp, np.float32)
    g = np.arange(size, dtype=np.float32)
    ex = np.exp(-(g - kp[..., 0:1] * (size - 1)) ** 2 / (2 * sigma ** 2))
    ey = np.exp(-(g - kp[..., 1:2] * (size - 1)) ** 2 / (2 * sigma ** 2))
    return ey[..., :, None] * ex[..., None, :] * joint_mask(kp)[..., None, None]


def heatmap_peaks(hm: np.ndarray) -> np.ndarray:
    """(14,H,W) -> (14,3) [x, y, peak] in normalized image coordinates."""
    Jn, H, W = hm.shape
    flat = hm.reshape(Jn, -1)
    i = flat.argmax(1)
    return np.stack([(i % W) / (W - 1), (i // W) / (H - 1), flat.max(1)], 1).astype(np.float32)


POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")


class CameraTeacher:
    """RGB frame -> (33,4) landmarks or None, using MediaPipe Pose (Tasks API, legacy fallback)."""
    def __init__(self, variant="full", model_dir="."):
        import mediapipe as mp
        self.mp, self.legacy, self._last = mp, None, -1
        try:
            from mediapipe.tasks.python import vision, BaseOptions
            path = os.path.join(model_dir, f"pose_landmarker_{variant}.task")
            if not os.path.exists(path):
                print(f"downloading MediaPipe pose model -> {path}")
                urllib.request.urlretrieve(POSE_MODEL_URL.format(v=variant), path)
            self.lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.VIDEO, num_poses=1))
        except Exception as e:
            if not hasattr(mp, "solutions"):
                raise
            print(f"MediaPipe Tasks API unavailable ({e}); using mp.solutions.pose")
            self.legacy = mp.solutions.pose.Pose()

    def __call__(self, rgb, t_ms):
        if self.legacy is not None:
            r = self.legacy.process(rgb)
            if not r.pose_landmarks:
                return None
            pts = r.pose_landmarks.landmark
        else:
            t_ms = max(int(t_ms), self._last + 1); self._last = t_ms
            r = self.lm.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), t_ms)
            if not r.pose_landmarks:
                return None
            pts = r.pose_landmarks[0]
        return np.array([[p.x, p.y, p.z, p.visibility] for p in pts], np.float32)


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def draw_skeleton(cv2, img, kp, color, thr=VIS_THR, thick=3):
    h, w = img.shape[:2]
    for a, b in SKELETON:
        if kp[a, 2] >= thr and kp[b, 2] >= thr:
            cv2.line(img, (int(kp[a, 0] * w), int(kp[a, 1] * h)), (int(kp[b, 0] * w), int(kp[b, 1] * h)),
                     color, thick)
    for x, y, v in kp:
        if v >= thr:
            cv2.circle(img, (int(x * w), int(y * h)), thick + 2, color, -1)


def heatmap_image(cv2, hm, size):
    """(14,H,W) -> colour image (max over joints)."""
    m = (np.clip(hm.max(0), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLORMAP_JET)


def heatmap_grid(cv2, hm, cell=64):
    """(14,H,W) -> 2x7 grid of per-joint maps with names."""
    tiles = []
    for j in range(J):
        t = cv2.applyColorMap(cv2.resize((np.clip(hm[j], 0, 1) * 255).astype(np.uint8), (cell, cell)),
                              cv2.COLORMAP_JET)
        cv2.putText(t, JOINTS[j][:8], (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        tiles.append(t)
    return np.vstack([np.hstack(tiles[:7]), np.hstack(tiles[7:])])
