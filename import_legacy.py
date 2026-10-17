"""
import_legacy.py - OPTIONAL: bring your earlier CSI-only recordings (data/<label>/<label>_<i>.npy
from record_dataset.py / csi_logger.py) into the new dataset format.

They have no camera and no per-sample timestamps, so they:
  * are assumed to be sampled at 100 Hz (the RX firmware's emit rate),
  * have no heatmap targets (the heatmap loss is masked out for them),
  * get session "legacy_<date>" from data/metadata.csv, so splits stay by session.

    python import_legacy.py --src ../data --out dataset

Label mapping: idle->idle, hands-crossed->hands_crossed, hands-up->hands_up,
arm-side->arm_side, arms-forward->arm_forward. Other labels are skipped.
"""
from __future__ import annotations
import argparse, csv, glob, json, os
import numpy as np
import common as C

MAP = {"idle": "idle", "hands-crossed": "hands_crossed", "hands-up": "hands_up",
       "arm-side": "arm_side", "arms-forward": "arm_forward", "arm-forward": "arm_forward",
       "hands_crossed": "hands_crossed", "hands_up": "hands_up", "arm_side": "arm_side",
       "arm_forward": "arm_forward"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data")
    ap.add_argument("--out", default="dataset")
    a = ap.parse_args()
    stamps = {}
    meta_csv = os.path.join(a.src, "metadata.csv")
    if os.path.exists(meta_csv):
        for row in csv.DictReader(open(meta_csv)):
            stamps[os.path.normpath(row["file"])] = row.get("timestamp", "")
    n = 0
    for f in sorted(glob.glob(os.path.join(a.src, "*", "*.npy"))):
        if "_raw" in f:
            continue
        label = os.path.basename(os.path.dirname(f))
        g = MAP.get(label)
        if g is None:
            print(f"skip {f}: label {label!r} is not one of the 5 classes"); continue
        amp = np.load(f).astype(np.float32)
        if amp.ndim != 2 or len(amp) < 150:
            print(f"skip {f}: shape {amp.shape}"); continue
        rel = os.path.normpath(os.path.relpath(f, a.src))
        ts = stamps.get(rel, "")
        session = "legacy_" + (ts[:10].replace("-", "") if ts else "unknown")
        rid = f"legacy_{os.path.splitext(os.path.basename(f))[0]}"
        d = os.path.join(a.out, g, rid)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "csi.npz"), amp=amp)
        json.dump({"gesture": g, "label": C.CLASS_TO_IDX[g], "session": session, "recording_id": rid,
                   "created": ts.replace(" ", "T") if ts else "", "legacy": True, "source": f,
                   "csi_samples": len(amp), "assumed_rate_hz": C.CSI_FS},
                  open(os.path.join(d, "meta.json"), "w"), indent=2)
        n += 1
    print(f"imported {n} legacy recordings into {a.out}/ (CSI only, no heatmaps)")


if __name__ == "__main__":
    main()
