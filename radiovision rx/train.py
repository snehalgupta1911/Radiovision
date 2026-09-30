"""
Training loop skeleton: student regresses the teacher's pose from CSI windows.
Loss/optimizer/features are intentionally minimal — the point is the shape of
the pipeline, not the final recipe. Iterate here alongside model.py.
"""
from __future__ import annotations
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import build_model


def frames_to_tensor(win, t: int, s_feat_fn) -> np.ndarray:
    """Turn a list[CSIFrame] into a fixed [T, C, S_feat] array. Pad/trim to T,
    single channel for one XIAO link. s_feat_fn maps a CSIFrame -> (S_feat,)."""
    feats = [s_feat_fn(f) for f in win][-t:]
    if not feats:
        raise ValueError("empty window")
    S = feats[0].shape[0]
    while len(feats) < t:
        feats.insert(0, np.zeros(S, np.float32))
    return np.stack(feats)[:, None, :]  # [T, C=1, S_feat]


class CSIPoseDataset(Dataset):
    def __init__(self, pairs, t, s_feat_fn):
        self.x = [frames_to_tensor(w, t, s_feat_fn) for w, _ in pairs]
        self.y = [p[:, :2] for _, p in pairs]  # keep x,y; drop visibility

    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])


def train(pairs, t, s_feat_fn, k=33, epochs=50, lr=1e-3, device="cpu"):
    ds = CSIPoseDataset(pairs, t, s_feat_fn)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    s_feat = ds.x[0].shape[-1]
    model = build_model("baseline", t=t, c=1, s_feat=s_feat, k=k, d=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        tot = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"epoch {ep:3d}  loss {tot/max(len(dl),1):.5f}")
    return model