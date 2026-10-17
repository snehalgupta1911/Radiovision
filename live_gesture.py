"""
live_gesture.py - live 5-class gesture prediction from RX CSI only (no camera).

    python live_gesture.py --port /dev/cu.usbmodem101 --model runs/compare/lambda_0.1/seed_0/model.pt
    python live_gesture.py --port ... --model ... --gui              # also show a window
    python live_gesture.py --port ... --model ... --gui --show-heatmap  # + predicted body heatmaps

Pipeline: RX serial CSI -> parse -> amplitude -> last window_sec on a 100 Hz grid ->
model (normalization stored inside) -> softmax -> optional smoothing -> gesture.
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
import torch
import common as C
import evaluate as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hz", type=float, default=10.0, help="predictions per second")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="exponential smoothing of probabilities (0 = none, 0.9 = very smooth)")
    ap.add_argument("--min-conf", type=float, default=0.5, help="below this, show 'uncertain'")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--show-heatmap", action="store_true")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until Ctrl+C / q)")
    a = ap.parse_args()

    device = C.pick_device()
    model, ck = E.load_model(a.model, device)
    win = ck["window_sec"]
    print(f"model {a.model}: lambda_heatmap={ck['lambda_heatmap']}, window {win}s, "
          f"test accuracy when trained {100 * ck['test_metrics']['accuracy']:.1f}%")

    reader = C.SerialCSIReader(a.port, a.baud)
    reader.start()
    print("waiting for CSI...")
    C.wait_for_csi(reader)
    if reader.n_sub != ck["n_sub"]:
        raise SystemExit(f"RX sends {reader.n_sub} subcarriers but the model expects {ck['n_sub']}")

    cv2 = None
    if a.gui:
        import cv2
        import heatmaps as HM
    probs_s = None
    last_label = None
    t_start = time.monotonic()
    next_t = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if a.seconds and now - t_start > a.seconds:
                break
            if now < next_t:
                time.sleep(min(0.01, next_t - now)); continue
            next_t += 1.0 / a.hz
            w = reader.live_window(now, win)
            if w is None:
                sys.stdout.write(f"\rwaiting for a full {win:.1f}s CSI window... ({reader.status()})   ")
                sys.stdout.flush(); continue
            x = torch.from_numpy(w[None].astype(np.float32)).to(device)
            with torch.no_grad():
                logits, hml = model(x, with_heatmaps=a.show_heatmap)
            p = torch.softmax(logits, 1)[0].cpu().numpy()
            probs_s = p if probs_s is None else a.smooth * probs_s + (1 - a.smooth) * p
            k = int(probs_s.argmax()); conf = float(probs_s[k])
            label = C.CLASSES[k] if conf >= a.min_conf else "uncertain"
            sys.stdout.write(f"\rPredicted gesture: {label:<14} Confidence: {conf:.2f}   "
                             f"[{' '.join(f'{c[:6]}={v:.2f}' for c, v in zip(C.CLASSES, probs_s))}]   ")
            sys.stdout.flush()
            if label != last_label:
                last_label = label

            if cv2 is not None:
                img = np.zeros((360, 640 if a.show_heatmap else 420, 3), np.uint8)
                C.put_text(cv2, img, label.upper().replace("_", " "), (16, 60), 1.4,
                           (0, 255, 0) if label != "uncertain" else (0, 200, 255), 3)
                C.put_text(cv2, img, f"confidence {conf:.2f}", (16, 100), 0.7)
                for i, (c, v) in enumerate(zip(C.CLASSES, probs_s)):
                    y = 140 + 40 * i
                    cv2.rectangle(img, (150, y - 18), (150 + int(240 * v), y + 4),
                                  (0, 200, 0) if i == k else (120, 120, 120), -1)
                    C.put_text(cv2, img, c, (16, y), 0.55)
                if a.show_heatmap and hml is not None:
                    hm = torch.sigmoid(hml)[0].cpu().numpy()
                    img[60:280, 410:630] = cv2.flip(HM.heatmap_image(cv2, hm, 220), 1)
                    C.put_text(cv2, img, "predicted body heatmaps", (410, 50), 0.45)
                C.put_text(cv2, img, reader.status(), (16, 350), 0.45, (255, 255, 0))
                cv2.imshow("CSI gesture (live, CSI only)", img)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    print()
    reader.stop()
    if cv2 is not None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
