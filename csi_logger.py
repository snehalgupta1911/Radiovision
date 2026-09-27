#!/usr/bin/env python3
"""
CSI Logger for Radiovision / RF-Pose Data Collection.

Connects to the ESP32-S3 over serial, records CSI data for a specified duration,
and saves labelled .npy datapoints compatible with the CSIMultiTaskModel and
CSIPreprocessor pipeline.

Usage
-----
    python csi_logger.py --label wave --duration 10 --port /dev/tty.usbmodem* --out data/

    # Interactive wizard (no args needed):
    python csi_logger.py

Output files per recording session
-----------------------------------
    data/<label>/<label>_<idx>_amp.npy     : amplitude (T, N_sub) float32
    data/<label>/<label>_<idx>_iq.npy      : raw I/Q  (T, N_sub, 2) int16
    data/<label>/<label>_<idx>_meta.json   : metadata (label, duration, fs, timestamp, …)
    data/labels.csv                        : running index  label,file,timestamp
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ─── Serial import (graceful fallback) ──────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ─── Internal parser ─────────────────────────────────────────────────────────
from rfpose.csi_esp32.esp32s3_parser import ESP32S3CSIParser
from rfpose.csi_esp32.preprocessor import CSIPreprocessor

# ─── Constants ───────────────────────────────────────────────────────────────
DEFAULT_BAUD      = 921600        # ESP32-S3 CSI log baud rate
DEFAULT_DURATION  = 10            # seconds per recording session
DEFAULT_OUT_DIR   = "data"
LABELS_CSV        = "labels.csv"
# Minimum window duration for gesture detection (model window = 100 frames @ 50 Hz = 2 s)
MIN_RECOMMENDED_DURATION = 5     # seconds – warn below this


# ═══════════════════════════════════════════════════════════════════════════════
#   Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def list_serial_ports() -> list:
    """Return a list of available serial port names."""
    if not HAS_SERIAL:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


def next_sample_index(out_dir: Path, label: str) -> int:
    """Scan existing files for the next available sample index."""
    label_dir = out_dir / label
    if not label_dir.exists():
        return 0
    existing = list(label_dir.glob(f"{label}_*_amp.npy"))
    if not existing:
        return 0
    indices = []
    for f in existing:
        parts = f.stem.split("_")
        # stem = <label>_<idx>_amp  →  parts[-2] is idx
        try:
            indices.append(int(parts[-2]))
        except (ValueError, IndexError):
            pass
    return max(indices) + 1 if indices else 0


def append_labels_csv(csv_path: Path, label: str, amp_file: Path, timestamp: str):
    """Append a row to the running labels.csv index."""
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "file", "timestamp"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "label":     label,
            "file":      str(amp_file),
            "timestamp": timestamp
        })


# ═══════════════════════════════════════════════════════════════════════════════
#   Core Recording Engine
# ═══════════════════════════════════════════════════════════════════════════════

class CSILogger:
    """
    Records raw CSI data from the ESP32-S3 serial stream for a fixed duration
    and saves labelled datapoints for model training.
    """

    def __init__(
        self,
        port:     str,
        baud:     int   = DEFAULT_BAUD,
        duration: float = DEFAULT_DURATION,
        out_dir:  str   = DEFAULT_OUT_DIR,
        label:    str   = "unknown",
        num_subcarriers: int = None,
        verbose:  bool  = True,
    ):
        self.port     = port
        self.baud     = baud
        self.duration = duration
        self.out_dir  = Path(out_dir)
        self.label    = label.strip().lower().replace(" ", "_")
        self.verbose  = verbose

        self.parser      = ESP32S3CSIParser(num_subcarriers=num_subcarriers)
        self.preprocessor = CSIPreprocessor(
            cutoff_freq  = 10.0,   # Hz  – removes high-freq noise above 10 Hz
            fs           = 50.0,   # Hz  – approximate ESP32-S3 CSI packet rate
            filter_order = 4,
            window_size  = 100,    # frames per model input window (2 s at 50 Hz)
            stride       = 25      # 50 % overlap between windows
        )

        if duration < MIN_RECOMMENDED_DURATION:
            print(
                f"[WARN] Duration {duration}s is shorter than the recommended minimum "
                f"({MIN_RECOMMENDED_DURATION}s). Gesture patterns may not be fully captured."
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(self) -> dict:
        """
        Open the serial port, record for `self.duration` seconds, parse, save, and return metadata.

        Returns
        -------
        dict with keys: label, amp_file, iq_file, meta_file, num_frames, num_windows, duration
        """
        if not HAS_SERIAL:
            raise RuntimeError(
                "pyserial is not installed. Run:  pip install pyserial"
            )

        label_dir = self.out_dir / self.label
        label_dir.mkdir(parents=True, exist_ok=True)

        sample_idx = next_sample_index(self.out_dir, self.label)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

        amp_file  = label_dir / f"{self.label}_{sample_idx}_amp.npy"
        iq_file   = label_dir / f"{self.label}_{sample_idx}_iq.npy"
        meta_file = label_dir / f"{self.label}_{sample_idx}_meta.json"

        amp_frames  = []
        iq_frames   = []
        rssi_frames = []

        if self.verbose:
            print(f"\n{'═'*60}")
            print(f"  CSI Logger  |  Label: '{self.label}'  |  #{sample_idx}")
            print(f"  Port: {self.port}  Baud: {self.baud}")
            print(f"  Recording for {self.duration} seconds …")
            print(f"{'═'*60}")
            print("  [Press Ctrl+C to stop early]\n")

        t_start = None

        try:
            with serial.Serial(self.port, self.baud, timeout=0.1) as ser:
                t_start = time.time()
                elapsed = 0.0

                while elapsed < self.duration:
                    raw_line = ser.readline()
                    if not raw_line:
                        elapsed = time.time() - t_start
                        continue

                    line = raw_line.decode("utf-8", errors="replace").strip()
                    amp, iq, rssi = self.parser.parse_line(line)

                    if amp is not None:
                        amp_frames.append(amp)
                        iq_frames.append(iq)
                        rssi_frames.append(rssi)

                        if self.verbose and len(amp_frames) % 50 == 0:
                            elapsed = time.time() - t_start
                            n_sub   = amp.shape[0]
                            print(
                                f"  [{elapsed:5.1f}s / {self.duration}s]  "
                                f"frames={len(amp_frames):5d}  "
                                f"subcarriers={n_sub}  "
                                f"RSSI={rssi} dBm"
                            )

                    elapsed = time.time() - t_start

        except KeyboardInterrupt:
            print("\n  [Stopped early by user]")
        except serial.SerialException as e:
            raise RuntimeError(f"Serial error on {self.port}: {e}") from e

        actual_duration = time.time() - t_start if t_start else 0.0

        if not amp_frames:
            print("[ERROR] No CSI frames captured. Check serial port, ESP32 firmware, and baud rate.")
            return {}

        # ── Assemble arrays ──────────────────────────────────────────────────
        min_sub   = min(a.shape[0] for a in amp_frames)
        amp_arr   = np.array([a[:min_sub] for a in amp_frames],  dtype=np.float32)   # (T, N_sub)
        iq_arr    = np.array([q[:min_sub] for q in iq_frames],   dtype=np.int16)     # (T, N_sub, 2)
        rssi_arr  = np.array(rssi_frames,                         dtype=np.int64)     # (T,)

        # ── Preprocessing & window count ─────────────────────────────────────
        windows_tensor = self.preprocessor.process(amp_arr)
        num_windows    = windows_tensor.shape[0]

        # ── Persist ──────────────────────────────────────────────────────────
        np.save(amp_file,  amp_arr)
        np.save(iq_file,   iq_arr)

        meta = {
            "label":           self.label,
            "sample_index":    sample_idx,
            "timestamp":       timestamp,
            "duration_s":      round(actual_duration, 3),
            "num_frames":      len(amp_frames),
            "num_subcarriers": min_sub,
            "num_windows":     num_windows,
            "approx_fps":      round(len(amp_frames) / max(actual_duration, 1e-3), 2),
            "port":            self.port,
            "baud":            self.baud,
            "amp_file":        str(amp_file),
            "iq_file":         str(iq_file),
        }

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        # ── Labels index ─────────────────────────────────────────────────────
        csv_path = self.out_dir / LABELS_CSV
        append_labels_csv(csv_path, self.label, amp_file, timestamp)

        if self.verbose:
            self._print_summary(meta)

        return meta

    # ── Pretty print ───────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(meta: dict):
        print(f"\n{'═'*60}")
        print(f"  ✅  Recording saved!")
        print(f"{'─'*60}")
        print(f"  Label        : {meta['label']} (sample #{meta['sample_index']})")
        print(f"  Duration     : {meta['duration_s']:.2f} s")
        print(f"  Frames       : {meta['num_frames']} @ ~{meta['approx_fps']:.1f} Hz")
        print(f"  Subcarriers  : {meta['num_subcarriers']}")
        print(f"  Model windows: {meta['num_windows']}  (window=100 frames, stride=25)")
        print(f"  Amp file     : {meta['amp_file']}")
        print(f"  IQ  file     : {meta['iq_file']}")
        print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#   Interactive Wizard
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_wizard() -> argparse.Namespace:
    """Prompt the user for required parameters interactively."""
    print("\n╔══════════════════════════════════════════╗")
    print("║    Radiovision CSI Data Logger Wizard    ║")
    print("╚══════════════════════════════════════════╝\n")

    # ── Gesture label ────────────────────────────────────────────────────────
    print("Available gesture labels:")
    suggested = [
        "wave", "swipe_left", "swipe_right", "push", "pull",
        "clap", "circle", "stand", "walk", "sit", "normal"
    ]
    for i, s in enumerate(suggested, 1):
        print(f"  {i:2d}. {s}")
    print("  Or type a custom label.")

    label_in = input("\nEnter label (number or name): ").strip()
    if label_in.isdigit() and 1 <= int(label_in) <= len(suggested):
        label = suggested[int(label_in) - 1]
    else:
        label = label_in if label_in else "unknown"

    # ── Duration ─────────────────────────────────────────────────────────────
    dur_in = input(f"\nRecording duration in seconds [{DEFAULT_DURATION}]: ").strip()
    try:
        duration = float(dur_in) if dur_in else DEFAULT_DURATION
    except ValueError:
        duration = DEFAULT_DURATION

    # ── Serial port ──────────────────────────────────────────────────────────
    ports = list_serial_ports()
    if ports:
        print(f"\nDetected serial ports:")
        for i, p in enumerate(ports, 1):
            print(f"  {i}. {p}")
        port_in = input(f"Select port (number or full path) [{ports[0]}]: ").strip()
        if port_in.isdigit() and 1 <= int(port_in) <= len(ports):
            port = ports[int(port_in) - 1]
        elif port_in:
            port = port_in
        else:
            port = ports[0]
    else:
        port = input("\nSerial port (e.g. /dev/tty.usbmodem1101): ").strip()

    # ── Output directory ─────────────────────────────────────────────────────
    out_in = input(f"\nOutput directory [{DEFAULT_OUT_DIR}]: ").strip()
    out_dir = out_in if out_in else DEFAULT_OUT_DIR

    return argparse.Namespace(
        label=label, duration=duration, port=port,
        baud=DEFAULT_BAUD, out=out_dir, num_subcarriers=None, quiet=False
    )


# ═══════════════════════════════════════════════════════════════════════════════
#   CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="csi_logger",
        description=(
            "Record labelled CSI data from the ESP32-S3 Wi-Fi sensor.\n"
            "If no arguments are given, an interactive wizard launches."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Record 10 s of a 'wave' gesture on macOS:
  python csi_logger.py --label wave --duration 10 --port /dev/tty.usbmodem1101

  # Record 15 s of standing still, save to a custom directory:
  python csi_logger.py --label stand --duration 15 --port COM4 --out ./dataset

  # Interactive wizard:
  python csi_logger.py
        """
    )
    p.add_argument("--label",    "-l", type=str,   default=None,
                   help="Gesture / activity label for this recording.")
    p.add_argument("--duration", "-d", type=float, default=None,
                   help=f"Recording duration in seconds (default: {DEFAULT_DURATION}).")
    p.add_argument("--port",     "-p", type=str,   default=None,
                   help="Serial port of the ESP32-S3 (e.g. /dev/tty.usbmodem1101 or COM4).")
    p.add_argument("--baud",     "-b", type=int,   default=DEFAULT_BAUD,
                   help=f"Serial baud rate (default: {DEFAULT_BAUD}).")
    p.add_argument("--out",      "-o", type=str,   default=DEFAULT_OUT_DIR,
                   help=f"Output directory for .npy datapoints (default: {DEFAULT_OUT_DIR}).")
    p.add_argument("--num-subcarriers", type=int,  default=None,
                   help="Expected subcarrier count (auto-detected if omitted).")
    p.add_argument("--quiet",    "-q", action="store_true",
                   help="Suppress verbose output.")
    return p


def main():
    parser = build_argparser()
    args   = parser.parse_args()

    # ── Interactive wizard if key args missing ────────────────────────────────
    if args.label is None or args.port is None:
        args = interactive_wizard()

    if args.duration is None:
        args.duration = DEFAULT_DURATION

    logger = CSILogger(
        port            = args.port,
        baud            = args.baud,
        duration        = args.duration,
        out_dir         = args.out,
        label           = args.label,
        num_subcarriers = args.num_subcarriers,
        verbose         = not args.quiet,
    )

    logger.record()


if __name__ == "__main__":
    main()
