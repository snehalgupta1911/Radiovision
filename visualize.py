"""
visualize.py - inspect a recording before trusting the model.

Interactive viewer (camera frame, keypoints, camera heatmaps, raw CSI, window target,
and - with --model - predicted heatmaps and gesture probabilities):
    python visualize.py --recording dataset/hands_up/s01_hands_up_20261009_101500
    python visualize.py --recording ... --model runs/compare/lambda_0.1/seed_0/model.pt
Keys: a/d = -/+ 0.1 s   j/l = -/+ 1 s   SPACE = play/pause   g = per-joint heatmaps
      s = save screenshot   q = quit

Synchronization check (wave your arms during a test recording, e.g.
`python collect.py --gesture idle --out sync_test ...`, then):
    python visualize.py --recording sync_test/idle/<id> --sync-check
It plots CSI motion energy against the camera's wrist speed and reports the time lag
that best aligns them. A lag near 0 (|lag| < 0.15 s) means CSI and camera are in sync.

Render one frame to a file without a window:
    python visualize.py --recording ... --time 5.0 --save frame.png
"""
from __future__ import annotations
import argparse, os
import numpy as np
import common as C
import heatmaps as HM
import dataset as D


def load_video(path):
    import cv2
    frames = []
    if os.path.exists(path):
        cap = cv2.VideoCapture(path)
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(f)
        cap.release()
    return frames


def csi_image(cv2, rec, width, height):
    """Whole-recording CSI amplitude (subcarriers x time), per-subcarrier normalized."""
    A = rec.G.astype(np.float32)
    A = (A - A.mean(0)) / (A.std(0) + 1e-3)
    img = np.clip((A.T + 3) / 6 * 255, 0, 255).astype(np.uint8)        # [S, M]
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)


def sync_check(rec, cam, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = rec.fs
    A = rec.G.astype(np.float32)
    k = int(0.2 * fs)
    lag_n = int(0.1 * fs)
    As = np.apply_along_axis(lambda c: np.convolve(c, np.ones(lag_n) / lag_n, mode="same"), 0, A)
    motion = np.zeros(len(A) - 1, np.float32)
    motion[lag_n - 1:] = np.abs(As[lag_n:] - As[:-lag_n]).mean(1)        # change over 0.1 s
    motion = np.convolve(motion, np.ones(k) / k, mode="same")
    tg = rec.grid[1:]
    ct, ok, joints = cam["host_t"], cam["ok"], cam["joints"]
    wr = joints[:, [4, 7], :2]                                          # wrists
    sp = np.linalg.norm(np.diff(wr, axis=0), axis=2).sum(1) / np.maximum(np.diff(ct), 1e-3)
    good = ok[1:] & ok[:-1]
    speed = np.interp(tg, ct[1:][good], sp[good]) if good.sum() > 10 else np.zeros_like(tg)
    speed = np.convolve(speed, np.ones(k) / k, mode="same")
    zm = (motion - motion.mean()) / (motion.std() + 1e-9)
    zs = (speed - speed.mean()) / (speed.std() + 1e-9)
    lags = np.arange(-int(fs), int(fs) + 1)
    xc = [np.mean(zm[max(0, l):len(zm) + min(0, l)] * zs[max(0, -l):len(zs) - max(0, l)]) for l in lags]
    best = lags[int(np.argmax(xc))] / fs
    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    t0 = tg[0]
    ax[0].plot(tg - t0, zm, label="CSI motion energy"); ax[0].plot(tg - t0, zs, label="camera wrist speed", alpha=0.8)
    ax[0].set_xlabel("time in recording (s)"); ax[0].legend(); ax[0].set_title("should rise and fall together")
    ax[1].plot(lags / fs, xc); ax[1].axvline(best, color="r")
    ax[1].set_xlabel("lag (s): positive = CSI reacts later than the camera")
    ax[1].set_title(f"cross-correlation, best lag = {best:+.2f} s, peak corr = {max(xc):.2f}")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return best, float(max(xc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--model")
    ap.add_argument("--sync-check", action="store_true")
    ap.add_argument("--time", type=float, default=None, help="start at this second")
    ap.add_argument("--save", help="render one frame to this png and exit")
    a = ap.parse_args()
    import cv2

    model, ck, device = None, None, None
    win = C.DEFAULT_WINDOW_SEC
    if a.model:
        import torch, evaluate as E
        device = C.pick_device()
        model, ck = E.load_model(a.model, device)
        win = ck["window_sec"]
    rec = D.Recording(a.recording, win, 0.1, trim_sec=0.0)
    cam_path = os.path.join(a.recording, "camera.npz")
    cam = np.load(cam_path) if os.path.exists(cam_path) else None
    frames = load_video(os.path.join(a.recording, "video.mp4"))
    print(f"{rec.rec_id}: gesture={rec.gesture} session={rec.session}  CSI {len(rec.t_csi)} samples "
          f"({len(rec.t_csi) / max(rec.t_csi[-1] - rec.t_csi[0], 1e-6):.1f} Hz), "
          f"{len(rec.ends)} windows, {int(rec.has_hm.sum())} with heatmap target, "
          f"camera frames {0 if cam is None else len(cam['host_t'])}, video frames {len(frames)}")

    if a.sync_check:
        if cam is None:
            raise SystemExit("no camera data in this recording")
        out = os.path.join(a.recording, "sync_check.png")
        lag, corr = sync_check(rec, cam, out)
        print(f"best lag {lag:+.2f} s (peak correlation {corr:.2f}) -> saved {out}")
        if corr < 0.2:
            print("correlation is weak: move your arms a lot during a sync-test recording")
        elif abs(lag) > 0.15:
            print("WARNING: CSI and camera look misaligned by more than 0.15 s")
        else:
            print("CSI and camera are in sync")
        return

    t_lo, t_hi = rec.grid[0], rec.grid[-1]
    t = t_lo + (a.time if a.time is not None else win)
    playing, show_grid = False, False
    W = 320 + 240 * (3 if model is None else 4)
    csi_img = csi_image(cv2, rec, W, 160)
    while True:
        t = float(np.clip(t, t_lo, t_hi))
        panels = []
        # camera frame + keypoints at time t
        fi = None
        if cam is not None and len(cam["host_t"]):
            fi = int(np.clip(np.searchsorted(cam["host_t"], t), 0, len(cam["host_t"]) - 1))
        frame = cv2.resize(frames[fi], (320, 240)) if (fi is not None and fi < len(frames)) else np.zeros((240, 320, 3), np.uint8)
        if fi is not None and cam["ok"][fi]:
            HM.draw_skeleton(cv2, frame, cam["joints"][fi], (0, 255, 0), thick=2)
        C.put_text(cv2, frame, "camera + keypoints", (6, 18), 0.45)
        panels.append(frame)
        # camera heatmap of this frame
        hm_f = cam["heatmaps"][fi].astype(np.float32) / 255 if fi is not None else np.zeros((HM.J, 32, 32))
        p = HM.heatmap_image(cv2, hm_f, 240); p = cv2.resize(p, (240, 240))
        C.put_text(cv2, p, "camera heatmaps (frame)", (6, 18), 0.45); panels.append(p)
        # window target (average over the window) and prediction
        e = int(np.clip(np.searchsorted(rec.grid, t), rec.T - 1, len(rec.grid) - 1))
        k = int(np.argmin(np.abs(rec.ends - e))) if len(rec.ends) else None
        tgt = rec.hm[k].astype(np.float32) / 255 if k is not None else np.zeros((HM.J, 32, 32))
        p = HM.heatmap_image(cv2, tgt, 240)
        C.put_text(cv2, p, "training target (window avg)", (6, 18), 0.45)
        if k is not None and not rec.has_hm[k]:
            C.put_text(cv2, p, "no target (too few poses)", (6, 40), 0.45, (0, 0, 255))
        panels.append(p)
        pred_txt = []
        hm_pred = None
        if model is not None:
            import torch
            x = torch.from_numpy(rec.G[e - rec.T + 1:e + 1][None].astype(np.float32)).to(device)
            with torch.no_grad():
                logits, hml = model(x)
            pr = torch.softmax(logits, 1)[0].cpu().numpy()
            hm_pred = torch.sigmoid(hml)[0].cpu().numpy()
            pp = HM.heatmap_image(cv2, hm_pred, 240)
            C.put_text(cv2, pp, "predicted heatmaps (CSI)", (6, 18), 0.45)
            panels.append(pp)
            pred_txt = [f"{c}: {v:.2f}" for c, v in zip(C.CLASSES, pr)]
            pred_txt.insert(0, f"PREDICTED: {C.CLASSES[int(pr.argmax())]}")
        top = np.hstack(panels)
        if top.shape[1] < W:
            top = np.hstack([top, np.zeros((240, W - top.shape[1], 3), np.uint8)])
        top = top[:, :W] if top.shape[1] > W else top
        # CSI strip with the current window marked
        strip = csi_img.copy()
        x1 = int((t - t_lo) / (t_hi - t_lo + 1e-9) * (W - 1)); x0 = int((t - win - t_lo) / (t_hi - t_lo + 1e-9) * (W - 1))
        cv2.rectangle(strip, (max(x0, 0), 0), (x1, 159), (255, 255, 255), 2)
        C.put_text(cv2, strip, "raw CSI amplitude (subcarrier x time), box = model input window", (6, 18), 0.45)
        info = np.zeros((130, W, 3), np.uint8)
        n_in = int(np.sum((rec.t_csi > t - win) & (rec.t_csi <= t)))
        C.put_text(cv2, info, f"label: {rec.gesture}   t = {t - t_lo:5.2f}s   CSI samples in window: {n_in}   "
                   f"camera frame #{fi}", (8, 24), 0.55, (0, 255, 255))
        for i, s in enumerate(pred_txt):
            C.put_text(cv2, info, s, (8 + 190 * (i // 3), 52 + 26 * (i % 3)), 0.5,
                       (0, 255, 0) if i == 0 else (220, 220, 220))
        C.put_text(cv2, info, "a/d 0.1s  j/l 1s  SPACE play  g joints  s save  q quit", (8, 124), 0.45, (150, 150, 150))
        img = np.vstack([top, strip, info])
        if show_grid:
            g = HM.heatmap_grid(cv2, hm_pred if hm_pred is not None else tgt, cell=W // 7)
            img = np.vstack([img, g[:, :W]])
        if a.save:
            cv2.imwrite(a.save, img); print(f"saved {a.save}"); return
        cv2.imshow("recording inspector", img)
        key = cv2.waitKey(33 if playing else 0) & 0xFF
        if playing:
            t += 1 / 30
        if key == ord("q"): break
        elif key == ord("a"): t -= 0.1
        elif key == ord("d"): t += 0.1
        elif key == ord("j"): t -= 1.0
        elif key == ord("l"): t += 1.0
        elif key == ord(" "): playing = not playing
        elif key == ord("g"): show_grid = not show_grid
        elif key == ord("s"):
            out = os.path.join(a.recording, f"inspect_{t - t_lo:.1f}s.png"); cv2.imwrite(out, img); print(f"saved {out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
