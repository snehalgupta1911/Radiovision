"""
Preprocessing: raw parsed CSI  ->  model-ready tensors + target heatmaps.

RF-Pose (the paper) feeds each encoder a clip of 100 complex RF-heatmap frames
(3.3 s @ 30 Hz), complex stored as two real channels, and supervises against
teacher keypoint *confidence maps* with per-pixel BCE. You have a single WiFi
link, so your "RF frame" is not a spatial map but a CSI vector over subcarriers.
The analog is a spectrogram tensor:

    input   x : [C, T, S]   C = feature channels (amp, [phase]),
                            T = time steps in the clip,
                            S = active subcarriers
    target  y : [K, H, W]   K keypoint gaussian confidence maps in image space

Key gotchas this module handles:
  * CSI arrives at an irregular rate -> resample onto a uniform time grid.
  * Variable CSI length per packet -> lock to one subcarrier count.
  * Per-subcarrier amplitude scale varies wildly -> standardize with saved stats.
  * Phase is corrupted -> linear-detrend (see csi.sanitize_phase).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .csi import CSIFrame, sanitize_phase


# ---------- 1. resample irregular CSI onto a uniform time grid ----------

def resample_to_grid(frames: list[CSIFrame], t_end: float, window_sec: float,
                     rate_hz: int, n_sub: int) -> np.ndarray | None:
    """Return a complex grid [T, n_sub] sampled uniformly over the clip, or None.
    Real and imaginary parts are interpolated separately (safe for phase wrap)."""
    frames = [f for f in frames if f.csi.shape[0] == n_sub and f.host_time is not None]
    if len(frames) < 2:
        return None
    frames.sort(key=lambda f: f.host_time)
    t_src = np.array([f.host_time for f in frames])
    stack = np.stack([f.csi for f in frames])            # [N, n_sub] complex
    T = int(round(window_sec * rate_hz))
    t_grid = np.linspace(t_end - window_sec, t_end, T)
    real = np.empty((T, n_sub), np.float32)
    imag = np.empty((T, n_sub), np.float32)
    for s in range(n_sub):
        real[:, s] = np.interp(t_grid, t_src, stack[:, s].real)
        imag[:, s] = np.interp(t_grid, t_src, stack[:, s].imag)
    return (real + 1j * imag).astype(np.complex64)


# ---------- 2. complex grid -> feature channels [C, T, S] ----------

def make_features(grid: np.ndarray, use_phase: bool = True) -> np.ndarray:
    amp = np.abs(grid).astype(np.float32)                 # [T, S]
    chans = [amp]
    if use_phase:
        ph = sanitize_phase(np.angle(grid))               # detrended [T, S]
        chans.append(ph.astype(np.float32))
    return np.stack(chans, axis=0)                        # [C, T, S]


@dataclass
class Normalizer:
    """Per-subcarrier standardization, fit on the TRAINING set only."""
    mean: np.ndarray  # [C, 1, S]
    std: np.ndarray   # [C, 1, S]

    @classmethod
    def fit(cls, feats: list[np.ndarray]) -> "Normalizer":
        x = np.stack(feats)                               # [N, C, T, S]
        mean = x.mean(axis=(0, 2), keepdims=True)[0]      # [C, 1, S]
        std = x.std(axis=(0, 2), keepdims=True)[0] + 1e-6
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


# ---------- 3. teacher keypoints -> target confidence maps [K, H, W] ----------

def keypoints_to_heatmaps(kps_norm: np.ndarray, H: int, W: int,
                          sigma: float = 1.5) -> np.ndarray:
    """kps_norm: [K, 2] in [0,1] image coords (x, y). Returns [K, H, W] with a
    gaussian bump at each visible keypoint. This is the RF-Pose output target."""
    K = kps_norm.shape[0]
    hm = np.zeros((K, H, W), np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    for k in range(K):
        x, y = kps_norm[k, 0] * (W - 1), kps_norm[k, 1] * (H - 1)
        if not (0 <= x <= W - 1 and 0 <= y <= H - 1):
            continue
        hm[k] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return hm


def heatmaps_to_keypoints(hm: np.ndarray) -> np.ndarray:
    """Inverse for inference/eval: argmax per channel -> [K, 2] normalized (x,y)."""
    K, H, W = hm.shape
    out = np.zeros((K, 2), np.float32)
    for k in range(K):
        j, i = np.unravel_index(np.argmax(hm[k]), (H, W))
        out[k] = [i / (W - 1), j / (H - 1)]
    return out


# ---------- 4. sliding windows over a recorded session ----------

def sliding_windows(n_items: int, win: int, stride: int):
    i = 0
    while i + win <= n_items:
        yield i, i + win
        i += stride