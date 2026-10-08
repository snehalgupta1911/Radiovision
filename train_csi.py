"""
train_csi.py - train a CNN + BiLSTM gesture classifier directly on the data/
folder written by record_dataset.py / csi_logger.py:

    data/<label>/<label>_<idx>.npy        amplitude, shape (n_frames, n_sub)

Each recording is cut into fixed-length windows. Validation holds out whole
recordings (20% per gesture), so the score reflects unseen recordings.

Usage (from ~/Radiovision-ESP32):
    python train_csi.py                       # uses ./data
    python train_csi.py --data data --win 96 --stride 24 --epochs 80
"""
import argparse, glob, json, os, time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class CSIGestureNet(nn.Module):
    def __init__(self, n_sub, n_classes, cnn_ch=64, lstm_hidden=64, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_sub, cnn_ch, 5, padding=2), nn.BatchNorm1d(cnn_ch), nn.ReLU(),
            nn.Conv1d(cnn_ch, cnn_ch, 5, padding=2), nn.BatchNorm1d(cnn_ch), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(cnn_ch, lstm_hidden, batch_first=True, bidirectional=True)
        self.fc_classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):                                   # x: [B, T, S]
        feat = self.cnn(x.permute(0, 2, 1))
        out, _ = self.bilstm(feat.permute(0, 2, 1))
        return self.fc_classifier(out.mean(dim=1))


def load_recordings(root):
    files = sorted(f for f in glob.glob(os.path.join(root, "*", "*.npy")) if "_raw" not in f)
    if not files:
        raise SystemExit(f"no recordings found in {root}/<label>/*.npy")
    recs = []
    for f in files:
        a = np.load(f).astype(np.float32)
        if a.ndim == 2 and a.shape[0] > 0:
            recs.append((os.path.basename(os.path.dirname(f)), f, a))
    sub_counts = Counter(a.shape[1] for _, _, a in recs)
    n_sub = sub_counts.most_common(1)[0][0]
    skipped = [f for _, f, a in recs if a.shape[1] != n_sub]
    recs = [r for r in recs if r[2].shape[1] == n_sub]
    print(f"found {len(files)} recordings, using {len(recs)} with {n_sub} subcarriers")
    if skipped:
        print(f"  skipped {len(skipped)} with other subcarrier counts {dict(sub_counts)}")
    return recs, n_sub


def make_windows(recs, win, stride):
    X, y, g = [], [], []
    classes = sorted({lab for lab, _, _ in recs})
    for ri, (lab, _, a) in enumerate(recs):
        if a.shape[0] < win:                                # pad short recordings
            a = np.vstack([np.repeat(a[:1], win - a.shape[0], 0), a])
        for s in range(0, a.shape[0] - win + 1, stride):
            X.append(a[s:s + win]); y.append(classes.index(lab)); g.append(ri)
    return np.stack(X), np.array(y, np.int64), np.array(g), classes


def split_by_recording(y, groups, val_frac, seed):
    """Per gesture, hold out ~val_frac of its recordings for validation."""
    rng = np.random.default_rng(seed)
    va_groups = []
    for c in np.unique(y):
        gs = np.unique(groups[y == c])
        rng.shuffle(gs)
        if len(gs) >= 2:
            va_groups += gs[:max(1, round(len(gs) * val_frac))].tolist()
    va = np.isin(groups, va_groups)
    return np.where(~va)[0], np.where(va)[0]


def augment(x):
    x = x * (1 + 0.1 * torch.randn(x.size(0), 1, 1, device=x.device))
    x = x + 0.05 * torch.randn_like(x)
    return torch.roll(x, shifts=int(torch.randint(-8, 9, (1,))), dims=1)


def run_epoch(model, loader, crit, device, opt=None):
    train = opt is not None
    model.train(train)
    tot, correct, n, P, Y = 0.0, 0, 0, [], []
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train: xb = augment(xb)
            logits = model(xb); loss = crit(logits, yb)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            pred = logits.argmax(1)
            tot += loss.item() * len(yb); correct += (pred == yb).sum().item(); n += len(yb)
            P.append(pred.cpu()); Y.append(yb.cpu())
    return tot / n, correct / n, torch.cat(P).numpy(), torch.cat(Y).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--win", type=int, default=96, help="frames per window")
    ap.add_argument("--stride", type=int, default=24, help="frames between windows")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gesture_model.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    recs, n_sub_raw = load_recordings(a.data)
    lens = [r[2].shape[0] for r in recs]
    print(f"frames per recording: min {min(lens)}, median {int(np.median(lens))}, max {max(lens)}")
    X, y, groups, classes = make_windows(recs, a.win, a.stride)
    tr, va = split_by_recording(y, groups, a.val, a.seed)

    print("\nper gesture:   recordings  windows(train/val)")
    for i, c in enumerate(classes):
        nrec = len(np.unique(groups[y == i]))
        print(f"  {c:<14} {nrec:6d}     {np.sum(y[tr] == i):5d} / {np.sum(y[va] == i):d}")

    mean_amp = X[tr].mean(axis=(0, 1))
    keep = mean_amp > 0.05 * np.median(mean_amp)            # drop null subcarriers
    X = X[:, :, keep]
    mu = X[tr].mean(axis=(0, 1), keepdims=True)
    sd = X[tr].std(axis=(0, 1), keepdims=True) + 1e-6
    X = (X - mu) / sd
    print(f"\nwindows X={X.shape}  active subcarriers {int(keep.sum())}/{n_sub_raw}")

    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    tr_dl = DataLoader(TensorDataset(Xt[tr], yt[tr]), batch_size=a.batch, shuffle=True)
    va_dl = DataLoader(TensorDataset(Xt[va], yt[va]), batch_size=a.batch)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = CSIGestureNet(X.shape[2], len(classes)).to(device)
    print(f"device={device}\n")

    counts = np.bincount(y[tr], minlength=len(classes)).clip(1)
    w = torch.tensor(len(tr) / (len(classes) * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    best, bad = -1.0, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tl, ta, _, _ = run_epoch(model, tr_dl, crit, device, opt)
        vl, vacc, _, _ = run_epoch(model, va_dl, crit, device)
        sched.step(vacc)
        flag = ""
        if vacc > best:
            best, bad, flag = vacc, 0, "  *saved*"
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "keep_mask": keep, "mean": mu, "std": sd, "win": a.win,
                        "n_sub_raw": n_sub_raw,
                        "model_kwargs": {"n_sub": int(X.shape[2]), "n_classes": len(classes)}},
                       a.out)
        else:
            bad += 1
        print(f"ep {ep:3d}  train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{vacc:.3f}  ({time.time()-t0:.1f}s){flag}")
        if bad >= a.patience:
            print("early stopping"); break

    ck = torch.load(a.out, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    _, acc, p, t = run_epoch(model, va_dl, crit, device)
    cm = np.zeros((len(classes), len(classes)), int)
    for ti, pi in zip(t, p): cm[ti, pi] += 1
    print(f"\nbest val acc = {acc:.3f}  (chance = {1/len(classes):.3f})   saved -> {a.out}")
    print("confusion matrix (rows = true, cols = predicted):")
    print(" " * 15 + "".join(f"{c[:12]:>13}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c[:14]:>14} " + "".join(f"{v:13d}" for v in row))
    json.dump({"classes": classes, "val_acc": acc, "win": a.win},
              open(a.out.replace(".pt", ".json"), "w"))


if __name__ == "__main__":
    main()