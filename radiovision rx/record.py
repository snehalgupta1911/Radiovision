"""
Synchronized capture: the heart of an RF-Pose-style dataset.

Reads the CSI serial stream and the webcam concurrently, stamps both with the
host clock (time.time()), and pairs each camera frame's teacher pose with the
CSI frames that fall in a window ending at that instant. Output is a stream of
(csi_window, pose_label) training examples with the camera used ONLY here.

This is a skeleton: threading + pairing logic are laid out; tune the window
length, buffering, and on-disk format to your model's input contract.
"""
from __future__ import annotations
import time, threading, collections, queue
import numpy as np

from .csi import parse_line, CSIFrame


class CSIReader(threading.Thread):
    """Background serial reader -> timestamped CSIFrames into a ring buffer."""
    def __init__(self, port: str, baud: int = 115200, maxlen: int = 20000):
        super().__init__(daemon=True)
        self.port, self.baud = port, baud
        self.buffer: collections.deque[CSIFrame] = collections.deque(maxlen=maxlen)
        self._stop = threading.Event()

    def run(self):
        import serial  # pip install pyserial
        ser = serial.Serial(self.port, self.baud, timeout=1)
        while not self._stop.is_set():
            line = ser.readline().decode("utf-8", "ignore")
            f = parse_line(line, host_time=time.time())
            if f is not None:
                self.buffer.append(f)

    def window_ending_at(self, t_end: float, duration: float) -> list[CSIFrame]:
        t_start = t_end - duration
        return [f for f in list(self.buffer) if t_start <= (f.host_time or 0) <= t_end]

    def stop(self): self._stop.set()


def record_session(port: str, teacher, window_sec: float = 0.5, cam_index: int = 0):
    """Yield (csi_window_frames, pose_label) pairs. Consumer decides encoding."""
    import cv2
    reader = CSIReader(port); reader.start()
    cap = cv2.VideoCapture(cam_index)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time()
            pose = teacher(frame)                       # camera -> label
            if pose is None:
                continue
            win = reader.window_ending_at(t, window_sec) # RF for same instant
            if not win:
                continue
            yield win, pose
    finally:
        cap.release(); reader.stop()