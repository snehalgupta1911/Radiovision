"""
RF-only inference: the payoff. No camera — pose predicted from CSI alone.
"""
from __future__ import annotations
import torch, time
from .record import CSIReader
from .train import frames_to_tensor


def run(model, port: str, t: int, s_feat_fn, window_sec: float = 0.5, device="cpu"):
    reader = CSIReader(port); reader.start()
    model.eval().to(device)
    try:
        while True:
            win = reader.window_ending_at(time.time(), window_sec)
            if len(win) < 2:
                time.sleep(0.02); continue
            x = torch.tensor(frames_to_tensor(win, t, s_feat_fn)).unsqueeze(0).to(device)
            with torch.no_grad():
                pose = model(x)[0].cpu().numpy()
            yield pose  # (K, 2) -> draw on a blank canvas, feed downstream, etc.
    finally:
        reader.stop()