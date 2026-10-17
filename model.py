"""
model.py - shared temporal CSI encoder with two heads.

    x [B, T, S]  (raw CSI amplitude, T = 100 steps = 1 s, S = 192 subcarriers)
      -> per-subcarrier normalization (stats from the training set, stored in the model)
      -> SHARED ENCODER
           Conv1d(S->128, k5) + GN + GELU     (subcarriers are channels, conv runs over time)
           Conv1d(128->128, k5) + GN + GELU, MaxPool(2)       T -> T/2
           Conv1d(128->128, k5) + GN + GELU, MaxPool(2)       T/2 -> T/4
           BiLSTM(128 -> 2x64)                               temporal context over the window
           mean-pool + max-pool over time -> z [B, 256]       = shared representation
      -> GESTURE HEAD:  z -> Linear 128 -> GELU -> Dropout -> Linear 5      logits [B, 5]
      -> HEATMAP HEAD:  z -> Linear 128*4*4 -> 3 x ConvTranspose (4->8->16->32) -> Conv 14
                        logits [B, 14, 32, 32]  (sigmoid -> joint confidence maps)

Both heads read the same z, so the heatmap loss's gradient flows back through z into
the shared encoder. With lambda_heatmap = 0 the heatmap head receives no gradient and
the encoder is trained by the gesture loss alone (= CSI-only baseline).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import heatmaps as HM


class SharedEncoder(nn.Module):
    def __init__(self, n_sub, ch=128, hidden=64, dropout=0.2):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv1d(cin, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.GELU())
        self.conv = nn.Sequential(block(n_sub, ch), block(ch, ch), nn.MaxPool1d(2), nn.Dropout(dropout),
                                  block(ch, ch), nn.MaxPool1d(2), nn.Dropout(dropout))
        self.lstm = nn.LSTM(ch, hidden, batch_first=True, bidirectional=True)
        self.out_dim = 4 * hidden

    def forward(self, x):                        # x: [B, T, S] normalized
        h = self.conv(x.transpose(1, 2))         # [B, ch, T/4]
        h, _ = self.lstm(h.transpose(1, 2))      # [B, T/4, 2*hidden]
        return torch.cat([h.mean(1), h.amax(1)], 1)


class GestureHead(nn.Module):
    def __init__(self, d, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, z):
        return self.net(z)


class HeatmapHead(nn.Module):
    def __init__(self, d, joints=HM.J):
        super().__init__()
        def up(a, b):
            return nn.Sequential(nn.ConvTranspose2d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.GELU())
        self.fc = nn.Sequential(nn.Linear(d, 128 * 16), nn.GELU())
        self.dec = nn.Sequential(up(128, 64), up(64, 32), up(32, 32), nn.Conv2d(32, joints, 3, padding=1))

    def forward(self, z):
        return self.dec(self.fc(z).view(-1, 128, 4, 4))    # [B, 14, 32, 32]


class CSIGestureNet(nn.Module):
    def __init__(self, n_sub, n_classes=5):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_sub))
        self.register_buffer("sd", torch.ones(n_sub))
        self.register_buffer("keep", torch.ones(n_sub))
        self.encoder = SharedEncoder(n_sub)
        self.gesture_head = GestureHead(self.encoder.out_dim, n_classes)
        self.heatmap_head = HeatmapHead(self.encoder.out_dim)

    def set_normalization(self, mu, sd, keep):
        self.mu.copy_(torch.as_tensor(mu)); self.sd.copy_(torch.as_tensor(sd))
        self.keep.copy_(torch.as_tensor(keep, dtype=torch.float32))

    def forward(self, x, with_heatmaps=True):
        x = (x - self.mu) / self.sd * self.keep
        z = self.encoder(x)
        logits = self.gesture_head(z)
        return (logits, self.heatmap_head(z)) if with_heatmaps else (logits, None)


def heatmap_loss(hm_logits, target, jmask, has_hm, pos_weight=20.0):
    """Masked, weighted per-pixel binary cross-entropy (as in RF-Pose).
    * only joints the teacher saw (jmask) in windows that have a target (has_hm) count
    * pixels inside the Gaussian blobs are up-weighted (1 + 20*target), otherwise the
      cheapest solution would be predicting empty maps
    Returns the mean over counted pixels (0 if the batch has no targets)."""
    m = (jmask * has_hm[:, None])[:, :, None, None]
    bce = F.binary_cross_entropy_with_logits(hm_logits, target, reduction="none")
    w = m * (1 + pos_weight * target)
    return (bce * w).sum() / w.sum().clamp(min=1.0)
