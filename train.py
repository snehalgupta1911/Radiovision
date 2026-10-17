"""
train.py - train the CSI gesture classifier with optional camera-heatmap auxiliary loss.

    L_total = L_gesture + lambda_heatmap * L_heatmap
        L_gesture = class-weighted cross-entropy over the 5 gestures
        L_heatmap = masked, weighted per-pixel BCE between predicted and camera heatmaps

Baseline (CSI -> gesture only):
    python train.py --lambda_heatmap 0 --name baseline
Auxiliary model:
    python train.py --lambda_heatmap 0.1 --name aux
Fair comparison in ONE command (same splits, same seeds, several lambdas):
    python train.py --lambda_heatmap 0 0.1 0.3 --seeds 0 1 2 --name compare
Leave-one-session-out cross-validation (needs >= 2 sessions):
    python train.py --lambda_heatmap 0 0.1 --cv --name cv

Splitting (no leakage): whole recordings only. The test set is a held-out session
(newest one, or --test-sessions); if there is only one session, ~20% of the recordings
of each class. From the remaining data one recording per class is kept as validation
for early stopping / model selection. Test data is never used for any decision.

Output: runs/<name>/
    summary.txt / summary.json        comparison table (mean +- std over seeds / folds)
    lambda_<L>/seed_<s>[/fold_<sess>]/
        model.pt                      trained model (use with live_gesture.py)
        history.csv                   per-epoch gesture / heatmap / total losses + val scores
        curves.png                    loss curves
        test_metrics.json, confusion.png
"""
from __future__ import annotations
import argparse, csv, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
import common as C
import dataset as D
import evaluate as E
from model import CSIGestureNet, heatmap_loss


def run_epoch_eval(model, recs, ids, device, lam):
    """Validation losses + metrics on whole recordings."""
    model.eval()
    items = D.items_of(recs, ids)
    gl = hl = 0.0; n = 0; yt, yp = [], []
    with torch.no_grad():
        for i in range(0, len(items), 128):
            x, y, hm, jm, hh = D.make_batch(recs, items[i:i + 128])
            logits, hml = model(torch.from_numpy(x).to(device))
            yt_ = torch.from_numpy(y).to(device)
            gl += F.cross_entropy(logits, yt_, reduction="sum").item()
            hl += heatmap_loss(hml, torch.from_numpy(hm).to(device), torch.from_numpy(jm).to(device),
                               torch.from_numpy(hh).to(device)).item() * len(y)
            n += len(y); yt += y.tolist(); yp += logits.argmax(1).cpu().tolist()
    m = E.classification_metrics(np.array(yt), np.array(yp))
    return gl / max(n, 1), hl / max(n, 1), m


def plot_curves(hist, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(ep, [h["train_gesture_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_gesture_loss"] for h in hist], label="val"); ax[0].set_title("gesture loss (CE)")
    ax[1].plot(ep, [h["train_heatmap_loss"] for h in hist], label="train")
    ax[1].plot(ep, [h["val_heatmap_loss"] for h in hist], label="val"); ax[1].set_title("heatmap loss (BCE)")
    ax[2].plot(ep, [h["train_total_loss"] for h in hist], label="train total")
    ax[2].plot(ep, [h["val_macro_f1"] for h in hist], label="val macro-F1")
    ax[2].plot(ep, [h["val_accuracy"] for h in hist], label="val accuracy"); ax[2].set_title("total loss / val scores")
    for a_ in ax:
        a_.set_xlabel("epoch"); a_.legend(fontsize=8); a_.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def train_one(recs, tr, va, te, lam, seed, a, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    S = recs[0].n_sub
    model = CSIGestureNet(S, len(C.CLASSES)).to(device)
    mu, sd, keep = D.norm_stats(recs, tr)
    model.set_normalization(mu, sd, keep)

    tr_items = D.items_of(recs, tr)
    counts = np.bincount([recs[i].label for i, _ in tr_items], minlength=len(C.CLASSES))
    present = counts > 0
    w = np.zeros(len(C.CLASSES), np.float32)
    w[present] = counts[present].sum() / (present.sum() * counts[present])      # inverse frequency
    class_w = torch.tensor(w, device=device)
    n_hm = sum(recs[i].has_hm[k] for i, k in tr_items)
    if lam > 0 and n_hm == 0:
        print("   WARNING: lambda > 0 but no training window has camera heatmaps (CSI-only recordings?)")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.05)
    hist, best, best_state, bad = [], -1.0, None, 0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time()
        perm = rng.permutation(len(tr_items))
        sg = sh = st = 0.0; nb = 0
        for b in range(0, len(perm), a.batch):
            items = [tr_items[j] for j in perm[b:b + a.batch]]
            x, y, hm, jm, hh = D.make_batch(recs, items, augment=True, rng=rng)
            x = torch.from_numpy(x).to(device); y = torch.from_numpy(y).to(device)
            logits, hml = model(x, with_heatmaps=lam > 0)
            lg = F.cross_entropy(logits, y, weight=class_w)
            if lam > 0:
                lh = heatmap_loss(hml, torch.from_numpy(hm).to(device), torch.from_numpy(jm).to(device),
                                  torch.from_numpy(hh).to(device))
            else:
                lh = torch.zeros((), device=device)        # baseline: heatmap head gets no gradient
            loss = lg + lam * lh
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sg += lg.item(); sh += lh.item(); st += loss.item(); nb += 1
        sched.step()
        eval_ids = va if va else tr
        vg, vh, vm = run_epoch_eval(model, recs, eval_ids, device, lam)
        h = {"epoch": ep, "train_gesture_loss": sg / nb, "train_heatmap_loss": sh / nb if lam > 0 else float("nan"),
             "train_total_loss": st / nb, "val_gesture_loss": vg, "val_heatmap_loss": vh,
             "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"]}
        hist.append(h)
        score = vm["macro_f1"] - 1e-3 * vg                   # tie-break on loss
        flag = ""
        if score > best:
            best, bad, flag = score, 0, " *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if a.verbose:
            print(f"   ep {ep:3d}  L_gesture {h['train_gesture_loss']:.3f}  L_heatmap {sh / nb:.3f}  "
                  f"L_total {h['train_total_loss']:.3f} | val L_g {vg:.3f} L_h {vh:.3f} acc {100 * vm['accuracy']:.0f}% "
                  f"F1 {100 * vm['macro_f1']:.0f}%  ({time.time() - t0:.1f}s){flag}")
        if va and bad >= a.patience:
            break
    model.load_state_dict(best_state)

    yt, yp, _, jerr = E.predict_recordings(model, recs, te, device)
    tm = E.classification_metrics(yt, yp)
    tm["heatmap_joint_error"] = jerr
    tm["epochs_trained"] = len(hist)
    torch.save({"state_dict": model.state_dict(), "n_sub": S, "classes": C.CLASSES,
                "window_sec": a.window_sec, "stride_sec": a.stride_sec, "fs": C.CSI_FS,
                "lambda_heatmap": lam, "seed": seed,
                "train_recordings": [recs[i].rec_id for i in tr + va],
                "test_recordings": [recs[i].rec_id for i in te], "test_metrics": tm},
               os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(hist[0])); wr.writeheader(); wr.writerows(hist)
    json.dump(tm, open(os.path.join(out_dir, "test_metrics.json"), "w"), indent=2)
    plot_curves(hist, os.path.join(out_dir, "curves.png"), f"lambda={lam} seed={seed}")
    E.plot_confusion(tm["confusion_matrix"], os.path.join(out_dir, "confusion.png"),
                     f"lambda={lam} seed={seed}  test acc {100 * tm['accuracy']:.0f}%")
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--lambda_heatmap", type=float, nargs="+", default=[0.0],
                    help="one or more values; 0 = CSI-only baseline")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--cv", action="store_true", help="leave-one-session-out cross-validation")
    ap.add_argument("--test-sessions", nargs="*")
    ap.add_argument("--window_sec", type=float, default=C.DEFAULT_WINDOW_SEC)
    ap.add_argument("--stride_sec", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    a = ap.parse_args()

    run_dir = os.path.join("runs", a.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    print(f"loading {a.data}/ (window {a.window_sec}s = {int(a.window_sec * C.CSI_FS)} samples, "
          f"stride {a.stride_sec}s)")
    recs = D.load_recordings(a.data, a.window_sec, a.stride_sec)
    device = C.pick_device()
    print(f"device: {device}\n")

    # ---- splits (identical for every lambda and seed) ----
    if a.cv:
        folds = D.cv_folds(recs)
        if len(folds) < 2:
            raise SystemExit("--cv needs at least 2 sessions")
        splits = []
        for test in folds:
            sess = recs[test[0]].session
            tr, va, te, _ = D.split(recs, test_sessions=[sess])
            splits.append((f"fold_{sess}", tr, va, te))
        print(f"leave-one-session-out: {len(splits)} folds")
    else:
        tr, va, te, how = D.split(recs, a.test_sessions)
        splits = [("", tr, va, te)]
        print(f"split: {how}")
        print(f"  train {len(tr)} recordings, val {len(va)} (early stopping), test {len(te)}")
    for _, tr, va, te in splits:
        missing = [C.CLASSES[c] for c in range(5) if not any(recs[i].label == c for i in te)]
        if missing:
            print(f"  note: test set has no recordings of {missing}")

    results = {}
    for lam in a.lambda_heatmap:
        for seed in a.seeds:
            for fold_name, tr, va, te in splits:
                tag = f"lambda={lam} seed={seed}" + (f" {fold_name}" if fold_name else "")
                print(f"=== {tag} ===")
                out = os.path.join(run_dir, f"lambda_{lam:g}", f"seed_{seed}", fold_name)
                tm = train_one(recs, tr, va, te, lam, seed, a, device, out)
                je = tm["heatmap_joint_error"]
                je_txt = "n/a" if (lam == 0 or np.isnan(je)) else f"{100 * je:.1f}% of image"
                print(f"   TEST: accuracy {100 * tm['accuracy']:.1f}%  balanced {100 * tm['balanced_accuracy']:.1f}%  "
                      f"macro-F1 {100 * tm['macro_f1']:.1f}%  (heatmap head joint error: {je_txt})\n")
                results.setdefault(lam, []).append(tm)

    # ---- comparison ----
    lines = [f"run {a.name}: {'leave-one-session-out CV' if a.cv else 'held-out test split'}, "
             f"seeds {a.seeds}, window {a.window_sec}s", ""]
    lines.append(f"{'lambda':>8}{'runs':>6}{'accuracy':>18}{'balanced acc':>18}{'macro-F1':>18}")
    summary = {}
    for lam, ms in results.items():
        acc = np.array([m["accuracy"] for m in ms]); bal = np.array([m["balanced_accuracy"] for m in ms])
        f1 = np.array([m["macro_f1"] for m in ms])
        lines.append(f"{lam:>8g}{len(ms):>6}{100 * acc.mean():>11.1f} +-{100 * acc.std():4.1f}"
                     f"{100 * bal.mean():>11.1f} +-{100 * bal.std():4.1f}{100 * f1.mean():>11.1f} +-{100 * f1.std():4.1f}")
        cm = np.sum([np.array(m["confusion_matrix"]) for m in ms], 0)
        pooled = E.classification_metrics(*_expand(cm))
        summary[str(lam)] = {"accuracy_mean": acc.mean(), "accuracy_std": acc.std(), "macro_f1_mean": f1.mean(),
                             "macro_f1_std": f1.std(), "balanced_accuracy_mean": bal.mean(),
                             "pooled": pooled}
        E.plot_confusion(cm, os.path.join(run_dir, f"confusion_lambda_{lam:g}.png"),
                         f"lambda={lam} pooled over {len(ms)} runs")
    lines.append("")
    lines.append("per-class F1 (pooled confusion matrix over all runs):")
    lines.append(f"{'lambda':>8}" + "".join(f"{c[:13]:>15}" for c in C.CLASSES))
    for lam in results:
        pc = summary[str(lam)]["pooled"]["per_class"]
        lines.append(f"{lam:>8g}" + "".join(f"{100 * pc[c]['f1']:>14.1f}%" for c in C.CLASSES))
    if len(results) > 1:
        lines.append("")
        lines.append("Read this carefully: a difference smaller than the +- spread across seeds/folds is not")
        lines.append("evidence that heatmap supervision helps (or hurts). Use more seeds / sessions / --cv.")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(run_dir, "summary.txt"), "w").write(txt + "\n")
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2, default=float)
    print(f"\nsaved {run_dir}/")


def _expand(cm):
    yt, yp = [], []
    for i in range(len(cm)):
        for j in range(len(cm)):
            yt += [i] * int(cm[i, j]); yp += [j] * int(cm[i, j])
    return np.array(yt, int), np.array(yp, int)


if __name__ == "__main__":
    main()
