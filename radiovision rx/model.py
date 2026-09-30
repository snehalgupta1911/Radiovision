"""
Student model: maps a window of CSI -> pose. THIS is the piece you iterate on.

Everything else in the package (capture, sync, teacher, training loop) is
agnostic to what goes here. Keep this contract stable and you can swap
architectures freely:

    forward(csi_window: Tensor[B, T, C, S_feat]) -> pose: Tensor[B, K, D]

    T       = time steps in the window (temporal context, RF-Pose uses a stack)
    C       = channels/links (1 for a single XIAO pair; grows with more antennas)
    S_feat  = per-subcarrier features (amplitude, and optionally sanitized phase)
    K, D    = keypoints x dims (33 x 2 for MediaPipe-Pose 2D)

Architecture menu to iterate through (all satisfy the contract above):
  - MLP baseline: flatten window -> a few dense layers. Sanity-check the rig.
  - CSI-as-image CNN: treat (subcarrier x time) as a 2D map -> 2D convs.
  - Temporal: 1D conv or LSTM/GRU over T, per-subcarrier encoder underneath.
  - RF-Pose-faithful: dual spatiotemporal encoders (amp / phase) + pose decoder
    emitting keypoint heatmaps; add a region-proposal head for multi-person.
  - Transformer: subcarriers/time as tokens with attention.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class StudentBaseline(nn.Module):
    """Deliberately dumb MLP so you can validate capture->train->infer end to
    end before investing in a real architecture. Replace me."""
    def __init__(self, t: int, c: int, s_feat: int, k: int, d: int = 2, hidden: int = 512):
        super().__init__()
        self.k, self.d = k, d
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(t * c * s_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, k * d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, C, S_feat]
        return self.net(x).view(x.size(0), self.k, self.d)


def build_model(name: str, **kw) -> nn.Module:
    return {"baseline": StudentBaseline}[name](**kw)