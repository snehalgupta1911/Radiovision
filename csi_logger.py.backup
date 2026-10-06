#!/usr/bin/env python3
"""
CSI Logger for ESP32 gesture-recognition data collection.

Reads `CSI_DATA,...` lines from the ESP32 receiver over serial, parses each
frame's I/Q pairs into per-subcarrier amplitude, and saves fixed-length,
gesture-labeled recordings ready for training.

Gestures: push, swipe-left, swipe-right, idle

USAGE
-----
    pip install pyserial numpy
    python csi_logger.py --port COM5          # Windows
    python csi_logger.py --port /dev/ttyUSB0  # Linux
    python csi_logger.py --port /dev/cu.usbserial-XXXX   # macOS

Then follow the on-screen menu: pick a gesture, choose how many reps, and
perform the gesture during each countdown window. Files land in ./data/.

OUTPUT
------
    data/<label>/<label>_<index>.npy     amplitude, shape (n_frames, n_sub)
    data/<label>/<label>_<index>_raw.npy raw I/Q, shape (n_frames, n_sub, 2)
    data/metadata.csv                    one row per recording
"""

import argparse
import csv
import os
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    import serial  # pyserial
except ImportError:
    sys.exit("Missing dependency. Run:  pip install pyserial numpy")

GESTURES = ["idle", "hands-up", "hands-crossed", "arm-side", "arms-forward"]


# --------------------------------------------------------------------------
# Serial reader: runs in a background thread, continuously parsing CSI frames
# into a timestamped buffer so recording windows can be sliced by time.
# --------------------------------------------------------------------------
class CSIReader(threading.Thread):
    def __init__(self, port, baud=115200):
        super().__init__(daemon=True)
        self.ser = serial.Serial(port, baud, timeout=1)
        self.lock = threading.Lock()
        self.buffer = deque(maxlen=20000)   # (t, amp[n_sub], iq[n_sub,2], rssi)
        self.running = True
        self.frame_count = 0
        self.n_sub = None

    @staticmethod
    def parse_line(line):
        """Parse CSI lines from the current RX firmware.
        
        Expected RX format:
        CSI,seq,timestamp,rssi,len,[I Q I Q I Q ...]
        
        Returns (amp, iq, rssi) or None.
        """
        if not line.startswith("CSI,"):
            return None

        line = line.strip()

        try:
            prefix, data = line.split("[", 1)
            data = data.rstrip("]")

            parts = prefix.rstrip(",").split(",")

            # CSI, seq, timestamp, rssi, len
            if len(parts) != 5:
                return None

            rssi = int(parts[3])
            length = int(parts[4])

            values = data.split()

            if len(values) != length:
                return None

            if length % 2 != 0:
                return None

            values = [int(v) for v in values]

            iq = np.array(
                list(zip(values[0::2], values[1::2])),
                dtype=np.int16
            )

        except (ValueError, IndexError):
            return None

        if len(iq) * 2 != length:
            return None

        amp = np.sqrt(
            iq[:, 0].astype(np.float32) ** 2 +
            iq[:, 1].astype(np.float32) ** 2
        )

        return amp, iq, rssi

    def run(self):
        while self.running:
            try:
                raw = self.ser.readline().decode("utf-8", errors="ignore")
            except Exception:
                continue
            parsed = self.parse_line(raw)
            if parsed is None:
                continue
            amp, iq, rssi = parsed
            with self.lock:
                if self.n_sub is None:
                    self.n_sub = amp.shape[0]
                self.buffer.append((time.time(), amp, iq, rssi))
                self.frame_count += 1

    def snapshot_since(self, t_start, t_end):
        """Return frames with t_start <= t <= t_end."""
        with self.lock:
            rows = [r for r in self.buffer if t_start <= r[0] <= t_end]
        return rows

    def current_fps(self, window=2.0):
        now = time.time()
        with self.lock:
            recent = [r for r in self.buffer if r[0] >= now - window]
        return len(recent) / window

    def stop(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Storage helpers
# --------------------------------------------------------------------------
def next_index(label_dir, label):
    os.makedirs(label_dir, exist_ok=True)
    existing = [f for f in os.listdir(label_dir)
                if f.startswith(label + "_") and f.endswith(".npy")
                and "_raw" not in f]
    return len(existing)


def save_recording(root, label, rows):
    label_dir = os.path.join(root, label)
    idx = next_index(label_dir, label)

    amp = np.stack([r[1] for r in rows])           # (n_frames, n_sub)
    iq = np.stack([r[2] for r in rows])            # (n_frames, n_sub, 2)
    rssi_mean = float(np.mean([r[3] for r in rows]))

    amp_path = os.path.join(label_dir, f"{label}_{idx}.npy")
    raw_path = os.path.join(label_dir, f"{label}_{idx}_raw.npy")
    np.save(amp_path, amp)
    np.save(raw_path, iq)

    meta_path = os.path.join(root, "metadata.csv")
    new_file = not os.path.exists(meta_path)
    with open(meta_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["label", "file", "n_frames", "n_sub",
                        "rssi_mean", "timestamp"])
        w.writerow([label, os.path.relpath(amp_path, root), amp.shape[0],
                    amp.shape[1], round(rssi_mean, 1),
                    time.strftime("%Y-%m-%d %H:%M:%S")])
    return amp_path, amp.shape


# --------------------------------------------------------------------------
# Interactive collection loop
# --------------------------------------------------------------------------
def countdown(msg, seconds):
    for s in range(seconds, 0, -1):
        print(f"\r{msg} {s}...", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 40, end="\r")


def record_one(reader, root, label, duration):
    countdown(f"Get ready for '{label}':", 3)
    print(f">>> GO! Perform '{label}' now  ({duration:.1f}s)")
    t0 = time.time()
    time.sleep(duration)
    t1 = time.time()

    rows = reader.snapshot_since(t0, t1)
    if len(rows) < 5:
        print(f"!! Only {len(rows)} frames captured — is the transmitter "
              f"running? Skipped.\n")
        return False
    path, shape = save_recording(root, label, rows)
    print(f"OK  saved {shape[0]} frames -> {path}\n")
    return True


def main():
    ap = argparse.ArgumentParser(description="ESP32 CSI gesture logger")
    ap.add_argument("--port", required=True, help="serial port of RECEIVER")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", default="data", help="output root dir")
    ap.add_argument("--duration", type=float, default=1.5,
                    help="seconds recorded per gesture rep")
    args = ap.parse_args()

    print(f"Opening {args.port} @ {args.baud} ...")
    reader = CSIReader(args.port, args.baud)
    reader.start()

    print("Warming up / checking link (3s)...")
    time.sleep(3)
    fps = reader.current_fps()
    if fps < 1:
        print("!! No CSI frames arriving. Check that the RECEIVER is on this "
              "port and the TRANSMITTER is powered.\n")
    else:
        print(f"Link OK — ~{fps:.0f} fps, {reader.n_sub} subcarriers.\n")

    try:
        while True:
            print("=" * 46)
            print("Gestures:")
            for i, g in enumerate(GESTURES):
                print(f"  {i+1}) {g}")
            print("  s) show live fps    q) quit")
            choice = input("Select gesture #, or command: ").strip().lower()

            if choice == "q":
                break
            if choice == "s":
                print(f"   live: ~{reader.current_fps():.0f} fps\n")
                continue
            if not choice.isdigit() or not (1 <= int(choice) <= len(GESTURES)):
                print("   ?? invalid choice\n")
                continue

            label = GESTURES[int(choice) - 1]
            reps_in = input(f"How many reps of '{label}'? [1]: ").strip()
            reps = int(reps_in) if reps_in.isdigit() else 1

            done = 0
            for r in range(reps):
                print(f"-- {label}: rep {r+1}/{reps} --")
                if record_one(reader, args.out, label, args.duration):
                    done += 1
                if r < reps - 1:
                    time.sleep(1.0)   # brief reset between reps
            print(f"Collected {done}/{reps} good reps of '{label}'.\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        reader.stop()
        print("Done. Data in ./" + args.out)


if __name__ == "__main__":
    main()
