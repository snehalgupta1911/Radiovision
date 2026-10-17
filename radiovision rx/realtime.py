"""
Realtime CSI pipeline for Radiovision: serial -> parse/validate -> rolling
buffer -> live amplitude heatmap + motion metric + MOTION/STATIC readout.

One code path, two sources:
  live    : python -m radiovision_ml.realtime --port /dev/ttyACM0
  replay  : python -m radiovision_ml.realtime --npz session1.npz --fps 60
  replay  : python -m radiovision_ml.realtime --file rawlog.txt   --fps 60

Replay drives the identical buffer + render logic, so it's both a no-hardware
demo and a debugging tool for the visualization itself.
"""
from __future__ import annotations
import argparse, threading, collections, time
import numpy as np

from .capture import parse_frame
from .analyze import amplitude, motion_metric, calibrate_auto


# ---------- thread-safe rolling buffer ----------

class FrameBuffer:
    def __init__(self, maxlen=600):
        self.buf = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.total = 0
        self.corrupt = 0

    def push_csi(self, csi):
        with self.lock:
            self.buf.append(csi); self.total += 1

    def mark_corrupt(self):
        with self.lock:
            self.corrupt += 1

    def snapshot(self):
        with self.lock:
            return list(self.buf), self.total, self.corrupt


# ---------- sources (background threads) ----------

class SerialSource(threading.Thread):
    def __init__(self, buffer, port, baud=115200):
        super().__init__(daemon=True); self.b = buffer; self.port = port; self.baud = baud; self._stop = threading.Event()
    def run(self):
        import serial
        ser = serial.Serial(self.port, self.baud, timeout=1)
        while not self._stop.is_set():
            line = ser.readline().decode('utf-8', 'ignore')
            if 'CSI,' not in line:
                continue
            f = parse_frame(line)
            if f is None: self.b.mark_corrupt()
            else:         self.b.push_csi(f[3])       # f = (seq, ts, rssi, csi)
    def stop(self): self._stop.set()

class ReplaySource(threading.Thread):
    """Replays frames from an .npz (csi array) or a raw text log at target fps."""
    def __init__(self, buffer, npz=None, file=None, fps=60):
        super().__init__(daemon=True); self.b = buffer; self.fps = fps; self._stop = threading.Event()
        if npz is not None:
            self.frames = list(np.load(npz)['csi'])
        else:
            self.frames = []
            for line in open(file, errors='ignore'):
                if 'CSI,' in line:
                    f = parse_frame(line)
                    if f is not None: self.frames.append(f[3])
    def run(self):
        dt = 1.0 / self.fps
        for csi in self.frames:
            if self._stop.is_set(): break
            self.b.push_csi(csi); time.sleep(dt)
    def stop(self): self._stop.set()


# ---------- render state (pure function, unit-testable) ----------

def freeze_null_mask(frames):
    """Decide once which subcarriers are active, so heatmap rows stay stable."""
    A = amplitude(np.stack(frames))
    med = np.median(A, axis=0)
    return med > np.percentile(med, 10)

def render_state(frames, keep_mask, window, maxlen):
    """Return (heatmap[S, maxlen], metric[maxlen], last_motion_value)."""
    A = amplitude(np.stack(frames))[:, keep_mask]        # [N, S]
    m = motion_metric(A, window)                          # [N]
    H = A.T                                               # [S, N]
    S, N = H.shape
    if N < maxlen:                                        # left-pad for a stable image
        H = np.hstack([np.zeros((S, maxlen-N), H.dtype), H])
        m = np.concatenate([np.zeros(maxlen-N, m.dtype), m])
    last = float(m[-window:].mean())
    return H, m, last


# ---------- live view ----------

def live(buffer, window=30, warmup=300, maxlen=600, redraw_ms=100, k=4.0):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    im = ax1.imshow(np.zeros((1, maxlen)), aspect='auto', origin='lower', cmap='viridis')
    ax1.set_ylabel('subcarrier'); title = ax1.set_title('CSI heatmap — waiting for data…')
    (line,) = ax2.plot(np.zeros(maxlen), lw=1.0, color='#1f77b4')
    thr_line = ax2.axhline(0, color='crimson', ls='--', lw=1)
    ax2.set_xlabel('frame (newest at right)'); ax2.set_ylabel('motion')
    state = {'mask': None, 'thr': None}

    def update(_):
        frames, total, corrupt = buffer.snapshot()
        n = len(frames)
        if n < 10:
            return im, line
        if state['mask'] is None and n >= min(warmup, maxlen//2):
            state['mask'] = freeze_null_mask(frames)
            m0 = motion_metric(amplitude(np.stack(frames))[:, state['mask']], window)
            state['thr'] = calibrate_auto(m0, k)          # baseline from quiet frames
            thr_line.set_ydata([state['thr'], state['thr']])
        if state['mask'] is None:
            title.set_text(f'CALIBRATING…  {n}/{warmup} frames'); return im, line

        H, m, last = render_state(frames, state['mask'], window, maxlen)
        im.set_data(H); im.set_clim(H.min(), H.max())
        im.set_extent([0, maxlen, 0, H.shape[0]])
        line.set_ydata(m); ax2.set_ylim(0, max(m.max()*1.1, state['thr']*1.5))
        moving = last > state['thr']
        yield_pct = 100*total/max(total+corrupt, 1)
        title.set_text(f"{'>>> MOTION <<<' if moving else 'static'}   "
                       f"metric={last:.3f} thr={state['thr']:.3f}   yield={yield_pct:.0f}%")
        title.set_color('crimson' if moving else 'black')
        return im, line

    _anim = FuncAnimation(fig, update, interval=redraw_ms, blit=False, cache_frame_data=False)
    plt.tight_layout(); plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port'); ap.add_argument('--npz'); ap.add_argument('--file')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--fps', type=int, default=60, help='replay speed')
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--maxlen', type=int, default=600, help='frames shown (~seconds*rate)')
    a = ap.parse_args()

    buf = FrameBuffer(maxlen=a.maxlen)
    if a.port:      src = SerialSource(buf, a.port, a.baud)
    elif a.npz:     src = ReplaySource(buf, npz=a.npz, fps=a.fps)
    elif a.file:    src = ReplaySource(buf, file=a.file, fps=a.fps)
    else:           ap.error('need --port, --npz, or --file')
    src.start()
    try:    live(buf, window=a.window, maxlen=a.maxlen)
    finally: src.stop()

if __name__ == '__main__':
    main()