"""
evaluate.py - metrics for the 5-class gesture task, and evaluation of a trained model.

Evaluate a saved model on recordings (e.g. a new session it has never seen):
    python evaluate.py --model runs/exp1/lambda_0.1/seed_0/model.pt --data dataset --sessions s05
Print the comparison table of a training run:
    python evaluate.py --summary runs/exp1

Metrics (per class and overall): accuracy, balanced accuracy (mean recall - fair under
class imbalance), precision, recall, F1, macro-F1, confusion matrix (rows = true class).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import common as C


def classification_metrics(y_true, y_pred, n=len(C.CLASSES)):
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    pred_n = cm.sum(0)
    precision = np.divide(tp, pred_n, out=np.zeros(n), where=pred_n > 0)
    recall = np.divide(tp, support, out=np.zeros(n), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) > 0)
    present = support > 0
    return {"accuracy": float(tp.sum() / max(cm.sum(), 1)),
            "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
            "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "per_class": {C.CLASSES[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                         "f1": float(f1[i]), "support": int(support[i])} for i in range(n)},
            "confusion_matrix": cm.tolist()}


def format_metrics(m, title=""):
    lines = [title] if title else []
    lines.append(f"accuracy {100 * m['accuracy']:.1f}%   balanced accuracy {100 * m['balanced_accuracy']:.1f}%   "
                 f"macro-F1 {100 * m['macro_f1']:.1f}%")
    lines.append(f"{'class':<14}{'precision':>10}{'recall':>8}{'F1':>8}{'support':>9}")
    for c in C.CLASSES:
        p = m["per_class"][c]
        lines.append(f"{c:<14}{100 * p['precision']:>9.1f}%{100 * p['recall']:>7.1f}%{100 * p['f1']:>7.1f}%{p['support']:>9}")
    cm = np.array(m["confusion_matrix"])
    lines.append("confusion matrix (rows = true, cols = predicted):")
    lines.append(" " * 14 + "".join(f"{c[:11]:>12}" for c in C.CLASSES))
    for c, row in zip(C.CLASSES, cm):
        lines.append(f"{c:<14}" + "".join(f"{v:>12d}" for v in row))
    return "\n".join(lines)


def plot_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.0f}%", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "black", fontsize=8)
    ax.set_xticks(range(len(C.CLASSES)), C.CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def predict_recordings(model, recs, ids, device, batch=128):
    """-> y_true, y_pred, probs, joint error info for windows of the given recordings."""
    import torch
    import dataset as D
    import heatmaps as HM
    model.eval()
    items = D.items_of(recs, ids)
    yt, yp, probs, jerr = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            x, y, hm, jm, hh = D.make_batch(recs, items[i:i + batch])
            logits, hml = model(torch.from_numpy(x).to(device))
            pr = torch.softmax(logits, 1).cpu().numpy()
            yt += y.tolist(); yp += pr.argmax(1).tolist(); probs.append(pr)
            hmp = torch.sigmoid(hml).cpu().numpy()
            for b in range(len(y)):
                if hh[b] > 0:
                    pk, tk = HM.heatmap_peaks(hmp[b]), HM.heatmap_peaks(hm[b])
                    m = jm[b] > 0
                    jerr.append(np.linalg.norm(pk[m, :2] - tk[m, :2], axis=1).mean())
    return (np.array(yt), np.array(yp), np.concatenate(probs) if probs else np.zeros((0, 5)),
            float(np.mean(jerr)) if jerr else float("nan"))


def load_model(path, device):
    import torch
    from model import CSIGestureNet
    ck = torch.load(path, map_location=device, weights_only=False)
    m = CSIGestureNet(ck["n_sub"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def print_summary(run_dir):
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    print(open(os.path.join(run_dir, "summary.txt")).read() if os.path.exists(os.path.join(run_dir, "summary.txt"))
          else json.dumps(s, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--data", default="dataset")
    ap.add_argument("--sessions", nargs="*", help="only these sessions (default: all)")
    ap.add_argument("--summary", help="run directory from train.py")
    a = ap.parse_args()
    if a.summary:
        print_summary(a.summary); return
    if not a.model:
        ap.error("--model or --summary required")
    import dataset as D
    device = C.pick_device()
    model, ck = load_model(a.model, device)
    recs = D.load_recordings(a.data, ck["window_sec"], ck["stride_sec"])
    ids = [i for i, r in enumerate(recs) if (not a.sessions or r.session in a.sessions) and r.n_sub == ck["n_sub"]]
    trained_on = set(ck.get("train_recordings", []))
    overlap = [recs[i].rec_id for i in ids if recs[i].rec_id in trained_on]
    if overlap:
        print(f"WARNING: {len(overlap)} of these recordings were used for training - scores will be optimistic")
    yt, yp, _, jerr = predict_recordings(model, recs, ids, device)
    m = classification_metrics(yt, yp)
    print(format_metrics(m, f"\n{a.model} on {len(ids)} recordings ({len(yt)} windows)"))
    if not np.isnan(jerr):
        print(f"heatmap head: mean joint error {100 * jerr:.1f}% of image (only meaningful if lambda > 0)")
    out = os.path.splitext(a.model)[0] + "_eval_confusion.png"
    plot_confusion(m["confusion_matrix"], out, os.path.basename(a.model))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
