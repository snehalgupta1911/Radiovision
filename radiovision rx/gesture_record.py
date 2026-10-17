"""
gesture_record.py  —  paired CSI + webcam gesture data recorder for Radiovision.

Self-contained: works flat in your folder (no package / __init__.py needed).
Reuses parse_frame from capture.py. Records labeled CSI windows for training a
gesture classifier. Webcam is a live viewfinder so you can time your gestures.

Controls (focus the webcam window):
  0-9   select the gesture class to record
  r     toggle recording on/off (while ON + a class selected, it auto-saves one
        window every WINDOW_SEC and shows a live count)
  q     quit and save everything to gestures.npz

Start with BIG motions (wave, raise arms, step, lean) — a single WiFi link can't
see finger shapes. Aim for ~100-300 samples per class, recorded across a few
different sessions/positions so the model generalizes.
"""
from __future__ import annotations
import argparse, threading, collections, time
import numpy as np
from capture import parse_frame          # capture.py is flat & self-contained

# ---- EDIT THIS: your gesture classes (start with large, distinct motions) ----
GESTURES = {
    "0": "idle",         # stand still
    "1": "wave",         # wave one arm side to side
    "2": "raise_arms",   # raise both arms overhead
    "3": "step",         # step in place / walk through the link
    "4": "push",         # push both hands toward the boards
}

WINDOW_SEC = 1.0     # seconds of CSI per sample
T          = 96      # time steps per sample (pad/truncate to this)


# ---- CSI reader thread (host-time-stamped rolling buffer) ----
class CSIReader(threading.Thread):
    def __init__(self, port, baud=115200, maxlen=4000):
        super().__init__(daemon=True)
        self.port, self.baud = port, baud
        self.buf = collections.deque(maxlen=maxlen)   # (host_time, csi)
        self.lock = threading.Lock(); self._stop = threading.Event()
    def run(self):
        import serial
        ser = serial.Serial(self.port, self.baud, timeout=1)
        while not self._stop.is_set():
            line = ser.readline().decode("utf-8", "ignore")
            if "CSI," not in line: continue
            f = parse_frame(line)
            if f is None: continue
            with self.lock:
                self.buf.append((time.time(), f[3]))   # f = (seq, ts, rssi, csi)
    def window(self, secs):
        now = time.time()
        with self.lock:
            return [csi for (t, csi) in self.buf if now - t <= secs]
    def stop(self): self._stop.set()


def window_to_features(frames, t=T):
    """list[complex csi] -> amplitude [t, n_sub], locked to the dominant length."""
    if len(frames) < 2: return None
    lens = collections.Counter(c.shape[0] for c in frames)
    n_sub = lens.most_common(1)[0][0]
    amp = np.abs(np.stack([c for c in frames if c.shape[0] == n_sub])).astype(np.float32)
    if amp.shape[0] >= t:
        amp = amp[-t:]
    else:                                   # left-pad short windows
        amp = np.vstack([np.zeros((t - amp.shape[0], n_sub), np.float32), amp])
    return amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--cam", type=int, default=0, help="0=built-in, 1=external")
    ap.add_argument("--out", default="gestures.npz")
    a = ap.parse_args()

    import cv2
    reader = CSIReader(a.port); reader.start()
    cap = cv2.VideoCapture(a.cam)
    X, y = [], []
    counts = {name: 0 for name in GESTURES.values()}
    label = None; recording = False; last_save = 0.0

    print("keys: 0-%d select class | r record on/off | q quit" % (len(GESTURES)-1))
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        status = f"class={label}  REC={'ON' if recording else 'off'}  " \
                 f"buf={len(reader.buf)}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if recording else (0, 200, 255), 2)
        y0 = 60
        for k, name in GESTURES.items():
            cv2.putText(frame, f"{k}:{name} [{counts[name]}]", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1); y0 += 25
        cv2.imshow("gesture recorder", frame)

        # auto-save one window per WINDOW_SEC while recording
        if recording and label is not None and time.time() - last_save >= WINDOW_SEC:
            feat = window_to_features(reader.window(WINDOW_SEC))
            if feat is not None:
                X.append(feat); y.append(label); counts[label] += 1
                last_save = time.time()

        key = cv2.waitKey(1) & 0xFF
        ch = chr(key) if key != 255 else ""
        if ch in GESTURES:
            label = GESTURES[ch]; print("class ->", label)
        elif ch == "r":
            recording = not recording; last_save = time.time()
            print("recording", "ON" if recording else "off")
        elif ch == "q":
            break

    cap.release(); cv2.destroyAllWindows(); reader.stop()
    if X:
        classes = sorted(set(y))
        yi = np.array([classes.index(v) for v in y], np.int64)
        np.savez(a.out, X=np.stack(X), y=yi, classes=np.array(classes))
        print(f"saved {len(X)} samples -> {a.out}  X{np.stack(X).shape}  "
              f"classes={classes}")
        for c in classes: print(f"  {c}: {counts[c]}")
    else:
        print("no samples recorded")

if __name__ == "__main__":
    main()