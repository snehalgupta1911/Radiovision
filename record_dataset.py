#!/usr/bin/env python3
"""
Batch CSI Recorder for Radiovision dataset collection.

Records multiple labelled sessions back-to-back with a countdown between each,
giving the operator time to set up the gesture before recording begins.

Usage
-----
    # Record 3 repetitions each of 5 gestures, 10 seconds each, 3-second countdown:
    python record_dataset.py --labels wave swipe_left push stand normal \\
                             --reps 3 --duration 10 --countdown 3 --port /dev/tty.usbmodem1101

    # Interactive:
    python record_dataset.py
"""

import argparse
import sys
import time
from pathlib import Path

from csi_logger import CSILogger, DEFAULT_BAUD, DEFAULT_DURATION, DEFAULT_OUT_DIR, list_serial_ports


GESTURE_LABELS = [
    "wave", "swipe_left", "swipe_right", "push", "pull",
    "clap", "circle", "stand", "walk", "sit", "normal"
]


def countdown(seconds: int, label: str):
    """Print a live countdown before recording starts."""
    print(f"\n  ┌─ Next: '{label}' ─────────────────────────────┐")
    for t in range(seconds, 0, -1):
        print(f"  │  Get ready … {t:2d}s                              │", end="\r")
        time.sleep(1)
    print(f"  │  ▶  RECORDING NOW!                            │")
    print(f"  └───────────────────────────────────────────────┘")


def interactive_batch_wizard() -> argparse.Namespace:
    print("\n╔═══════════════════════════════════════════════════╗")
    print("║   Radiovision Batch Dataset Collection Wizard    ║")
    print("╚═══════════════════════════════════════════════════╝\n")

    print("Available labels:")
    for i, g in enumerate(GESTURE_LABELS, 1):
        print(f"  {i:2d}. {g}")
    raw = input("\nEnter label numbers or names (space-separated, e.g. '1 2 stand'): ").strip()
    labels = []
    for tok in raw.split():
        if tok.isdigit() and 1 <= int(tok) <= len(GESTURE_LABELS):
            labels.append(GESTURE_LABELS[int(tok) - 1])
        elif tok:
            labels.append(tok.lower().replace(" ", "_"))
    if not labels:
        labels = ["wave"]

    reps_in = input(f"\nRepetitions per label [3]: ").strip()
    reps = int(reps_in) if reps_in.isdigit() else 3

    dur_in = input(f"Recording duration per session in seconds [{DEFAULT_DURATION}]: ").strip()
    try:
        duration = float(dur_in) if dur_in else DEFAULT_DURATION
    except ValueError:
        duration = DEFAULT_DURATION

    cd_in = input("Countdown before each recording in seconds [3]: ").strip()
    try:
        cd = int(cd_in) if cd_in else 3
    except ValueError:
        cd = 3

    ports = list_serial_ports()
    if ports:
        print(f"\nDetected ports: {ports}")
        port_in = input(f"Select port [{ports[0]}]: ").strip()
        port = ports[int(port_in) - 1] if port_in.isdigit() and 1 <= int(port_in) <= len(ports) else (port_in or ports[0])
    else:
        port = input("\nSerial port: ").strip()

    out_in = input(f"\nOutput directory [{DEFAULT_OUT_DIR}]: ").strip()
    out_dir = out_in if out_in else DEFAULT_OUT_DIR

    return argparse.Namespace(
        labels=labels, reps=reps, duration=duration,
        countdown=cd, port=port, baud=DEFAULT_BAUD,
        out=out_dir, quiet=False
    )


def main():
    p = argparse.ArgumentParser(
        prog="record_dataset",
        description="Batch CSI dataset recorder for Radiovision gesture sensing."
    )
    p.add_argument("--labels",    nargs="+", default=None,    help="Gesture labels to record.")
    p.add_argument("--reps",      type=int,  default=3,       help="Repetitions per label.")
    p.add_argument("--duration",  type=float,default=DEFAULT_DURATION,  help="Seconds per session.")
    p.add_argument("--countdown", type=int,  default=3,       help="Countdown before each session.")
    p.add_argument("--port",  "-p", type=str, default=None,   help="Serial port.")
    p.add_argument("--baud",  "-b", type=int, default=DEFAULT_BAUD,    help="Baud rate.")
    p.add_argument("--out",   "-o", type=str, default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument("--quiet", "-q", action="store_true")
    args = p.parse_args()

    if args.labels is None or args.port is None:
        args = interactive_batch_wizard()

    total    = len(args.labels) * args.reps
    done     = 0
    failures = []

    print(f"\n{'━'*60}")
    print(f"  Batch Recording Plan")
    print(f"  Labels  : {args.labels}")
    print(f"  Reps    : {args.reps} per label  ({total} total)")
    print(f"  Duration: {args.duration}s per session")
    print(f"  Port    : {args.port}")
    print(f"{'━'*60}\n")

    for label in args.labels:
        for rep in range(args.reps):
            print(f"\n  Session {done+1}/{total}  —  '{label}' rep {rep+1}/{args.reps}")
            countdown(args.countdown, label)

            logger = CSILogger(
                port     = args.port,
                baud     = args.baud,
                duration = args.duration,
                out_dir  = args.out,
                label    = label,
                verbose  = not args.quiet,
            )
            try:
                meta = logger.record()
                if meta:
                    done += 1
                else:
                    failures.append((label, rep))
            except Exception as e:
                print(f"  [ERROR] {e}")
                failures.append((label, rep))

    print(f"\n{'━'*60}")
    print(f"  Batch complete!  {done}/{total} sessions saved.")
    if failures:
        print(f"  Failed sessions: {failures}")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()
