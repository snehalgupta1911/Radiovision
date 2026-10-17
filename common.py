"""
common.py - constants and the CSI serial reader shared by every script.

Reads the EXISTING RX firmware output unchanged. Each CSI line looks like
    CSI,<seq>,<timestamp_us>,<rssi>,<len>,[v0 v1 v2 ... v(len-1)]
where v are signed bytes, interleaved (imag, real) per subcarrier, so a line
with len = 384 carries 192 subcarriers. Any other line ("RX starting...",
"Connecting to TX....", "Connected! Channel:6") is ignored, and malformed or
truncated CSI lines are counted and dropped.
"""
from __future__ import annotations
import collections, re, threading, time
import numpy as np

# ---------------------------------------------------------------------------
# The five gesture classes (order = label index). Do not change.
# ---------------------------------------------------------------------------
CLASSES = ["idle", "hands_crossed", "hands_up", "arm_side", "arm_forward"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Expected CSI rate: the RX firmware emits at most EMIT_HZ = 100 lines/s.
CSI_FS = 100.0            # Hz, the uniform grid every recording is resampled to
DEFAULT_WINDOW_SEC = 1.0  # 1.0 s at 100 Hz = 100 CSI samples per window
MAX_GAP_SEC = 0.10        # a window is discarded if CSI is missing for longer than this

CSI_LINE = re.compile(r"CSI,(\d+),(\d+),(-?\d+),(\d+),\[([^\]]*)\]")


def parse_csi_line(line: str):
    """-> (seq, ts_us, rssi, values int8 array) or None if the line is not a valid CSI record."""
    m = CSI_LINE.search(line)
    if not m:
        return None
    try:
        seq, ts, rssi, n = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        vals = np.array(m.group(5).split(), dtype=np.int16)
    except ValueError:
        return None
    if n == 0 or len(vals) != n or n % 2 or vals.min(initial=0) < -128 or vals.max(initial=0) > 127:
        return None
    return seq, ts, rssi, vals.astype(np.int8)


def csi_amplitude(vals_int8: np.ndarray) -> np.ndarray:
    """int8 [..., 2*S] interleaved (imag, real) -> amplitude [..., S] = sqrt(I^2 + Q^2)."""
    v = vals_int8.astype(np.float32)
    return np.sqrt(v[..., 0::2] ** 2 + v[..., 1::2] ** 2)


class SerialCSIReader(threading.Thread):
    """Background thread reading the RX board.

    * every valid CSI line is stamped with the Mac's monotonic clock on arrival
    * keeps a rolling live buffer (for live inference)
    * when .recording is True, also keeps every raw record (for saving)
    """
    def __init__(self, port, baud=115200, keep_sec=5.0):
        super().__init__(daemon=True)
        self.port, self.baud, self.keep = port, baud, keep_sec
        self.lock = threading.Lock()
        self.live_t, self.live_amp = collections.deque(), collections.deque()
        self.rows = []
        self.recording = False
        self.n_ok = self.n_bad = self.n_other = 0
        self.len_counts = collections.Counter()
        self.n_sub = None
        self.rate = 0.0
        self.last_text = ""
        self.error = None
        self._stop = threading.Event()

    def run(self):
        try:
            import serial
            ser = serial.Serial(self.port, self.baud, timeout=0.05)
            ser.reset_input_buffer()
        except Exception as e:
            self.error = str(e); return
        buf = b""
        n_rate, t_rate = 0, time.monotonic()
        while not self._stop.is_set():
            try:
                chunk = ser.read(max(1, ser.in_waiting))
            except Exception as e:
                self.error = str(e); break
            if chunk:
                buf += chunk
                *lines, buf = buf.split(b"\n")
                now = time.monotonic()
                for raw in lines:
                    line = raw.decode("ascii", "ignore").strip()
                    if not line:
                        continue
                    if "CSI," not in line:
                        self.n_other += 1; self.last_text = line[:80]; continue
                    rec = parse_csi_line(line)
                    if rec is None:
                        self.n_bad += 1; continue
                    seq, ts, rssi, vals = rec
                    self.n_ok += 1; n_rate += 1
                    self.len_counts[len(vals)] += 1
                    if self.n_sub is None and self.n_ok >= 30:
                        self.n_sub = self.len_counts.most_common(1)[0][0] // 2
                    with self.lock:
                        if self.recording:
                            self.rows.append((now, seq, ts, rssi, vals))
                        if self.n_sub is not None and len(vals) == 2 * self.n_sub:
                            self.live_t.append(now); self.live_amp.append(csi_amplitude(vals))
                            while self.live_t and now - self.live_t[0] > self.keep:
                                self.live_t.popleft(); self.live_amp.popleft()
                if len(buf) > 65536:          # no newline for a long time: garbage
                    buf = b""
            now = time.monotonic()
            if now - t_rate >= 1.0:
                self.rate, n_rate, t_rate = n_rate / (now - t_rate), 0, now
        ser.close()

    def stop(self):
        self._stop.set()

    def start_recording(self):
        with self.lock:
            self.rows = []; self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False
            rows, self.rows = self.rows, []
        return rows

    def n_recorded(self):
        with self.lock:
            return len(self.rows)

    def live_window(self, t_end, window_sec, fs=CSI_FS, max_gap=MAX_GAP_SEC):
        """Amplitude on a uniform grid ending at t_end -> [T, S], or None if data is missing."""
        with self.lock:
            if len(self.live_t) < 4:
                return None
            t = np.fromiter(self.live_t, float, len(self.live_t))
            A = np.stack(self.live_amp)
        T = int(round(window_sec * fs))
        grid = t_end - (T - 1 - np.arange(T)) / fs
        idx = np.searchsorted(t, grid, side="right") - 1
        if idx[0] < 0 or np.max(grid - t[idx]) > max_gap:
            return None
        return A[idx]

    def status(self):
        if self.error:
            return f"SERIAL ERROR: {self.error}"
        return (f"CSI {self.rate:5.1f} Hz  ok={self.n_ok}  malformed={self.n_bad}  "
                f"subcarriers={self.n_sub}")


def wait_for_csi(reader, timeout=20):
    t0 = time.monotonic()
    while reader.n_sub is None:
        time.sleep(0.2)
        if reader.error or time.monotonic() - t0 > timeout:
            raise SystemExit(f"No CSI from the RX board ({reader.status()}).\n"
                             f"Last non-CSI line: {reader.last_text!r}\n"
                             "Check the port (ls /dev/cu.usbmodem*), that TX is powered, and that no "
                             "Serial Monitor has the port open.")


def save_csi_rows(rows, path):
    """Save raw records: arrival time, seq, radio timestamp, rssi, raw int8 values."""
    n = len(rows)
    L = max((len(r[4]) for r in rows), default=0)
    csi = np.zeros((n, L), np.int8)
    lens = np.zeros(n, np.int16)
    for i, r in enumerate(rows):
        csi[i, :len(r[4])] = r[4]; lens[i] = len(r[4])
    np.savez_compressed(path, host_t=np.array([r[0] for r in rows], np.float64),
                        seq=np.array([r[1] for r in rows], np.uint32),
                        ts_us=np.array([r[2] for r in rows], np.uint32),
                        rssi=np.array([r[3] for r in rows], np.int8), csi=csi, csi_len=lens)


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def put_text(cv2, img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
