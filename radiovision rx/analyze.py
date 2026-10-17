"""
CSI analysis + motion detection for Radiovision.

Answers the bring-up question: "is the channel actually changing when someone
moves, or am I capturing static noise?" Produces:
  1. an amplitude heatmap (subcarrier x time) -- the single-link analog of the
     RF-Pose heatmap, and the thing you eyeball to see motion,
  2. a motion metric time series with a calibrated threshold, so the decision
     "change is occurring" is a number, not a vibe.

Load a session saved by capture.py (.npz with keys: ts, csi[N, n_sub] complex).
"""
from __future__ import annotations
import argparse
import numpy as np


# ---------- load + clean ----------

def load_session(path):
    d = np.load(path)
    return d['ts'], d['csi']                 # ts[N], csi[N, n_sub] complex

def amplitude(csi):
    return np.abs(csi).astype(np.float32)    # [N, n_sub]

def drop_null_subcarriers(A, frac=0.9):
    """Remove subcarriers that are ~zero most of the time (guard/DC/pilot nulls).
    Keeps columns whose median amplitude is above a small fraction of the max."""
    med = np.median(A, axis=0)
    keep = med > (frac * 0.05 * med.max() + 1e-6)   # conservative
    keep = med > np.percentile(med, 10)             # drop the quietest 10%
    return A[:, keep], keep


# ---------- motion metric ----------

def motion_metric(A, window=30):
    """Windowed temporal variance of amplitude, averaged over subcarriers.

    For each subcarrier we measure how much its amplitude wobbles over the last
    `window` frames, then average across subcarriers. A still channel -> low and
    flat. A person moving -> the multipath shifts -> variance jumps. This is the
    standard CSI motion indicator and is robust to single-frame glitches.

    Returns m[N] aligned to frame index (first `window-1` frames are warm-up).
    """
    N, S = A.shape
    # per-subcarrier z-score so loud subcarriers don't dominate the average
    A = (A - A.mean(0)) / (A.std(0) + 1e-6)
    m = np.zeros(N, np.float32)
    for t in range(N):
        lo = max(0, t - window + 1)
        m[t] = A[lo:t+1].var(axis=0).mean()
    return m

def frame_delta(A):
    """Instantaneous change: mean |A[t]-A[t-1]| across subcarriers. Spikes on
    sudden movement; noisier than the windowed metric, good as a companion."""
    d = np.zeros(A.shape[0], np.float32)
    d[1:] = np.abs(np.diff(A, axis=0)).mean(axis=1)
    return d


# ---------- decision threshold ----------

def calibrate(metric, baseline_slice, k=4.0):
    """Threshold from a STATIC baseline period (record ~5 s of empty room first).
    Anything above mean + k*std of the baseline counts as 'change occurring'."""
    b = metric[baseline_slice]
    thr = b.mean() + k * b.std()
    return float(thr)

def calibrate_auto(metric, k=4.0):
    """Fallback when you have no labelled baseline: assume the quietest 20% of
    the recording is 'static' and threshold off that. Less reliable than a real
    empty-room baseline."""
    q = np.quantile(metric, 0.20)
    quiet = metric[metric <= q]
    return float(quiet.mean() + k * quiet.std())


# ---------- report + plot ----------

def summarize(metric, thr):
    frac = float((metric > thr).mean())
    contrast = float(metric.max() / (np.median(metric) + 1e-9))
    print(f"threshold            : {thr:.4f}")
    print(f"frames over threshold: {100*frac:.1f}%")
    print(f"peak / median ratio  : {contrast:.1f}x  "
          f"({'clear motion signal' if contrast > 3 else 'WEAK — check antennas/placement'})")
    return frac, contrast

def plot(A, metric, thr, out='csi_report.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.imshow(A.T, aspect='auto', origin='lower', cmap='viridis',
               extent=[0, A.shape[0], 0, A.shape[1]])
    ax1.set_ylabel('subcarrier'); ax1.set_title('CSI amplitude heatmap')
    ax2.plot(metric, lw=1.0, color='#1f77b4', label='motion metric')
    ax2.axhline(thr, color='crimson', ls='--', lw=1, label='threshold')
    ax2.fill_between(np.arange(len(metric)), 0, metric.max(),
                     where=metric > thr, color='crimson', alpha=0.12,
                     label='change detected')
    ax2.set_xlabel('frame'); ax2.set_ylabel('motion'); ax2.legend(loc='upper right', fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz'); ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--baseline', type=int, default=0,
                    help='num initial frames that are static (0 = auto-calibrate)')
    ap.add_argument('--out', default='csi_report.png')
    a = ap.parse_args()
    ts, csi = load_session(a.npz)
    A, keep = drop_null_subcarriers(amplitude(csi))
    print(f"frames={A.shape[0]}  active subcarriers={A.shape[1]} (of {csi.shape[1]})")
    m = motion_metric(A, a.window)
    thr = calibrate(m, slice(0, a.baseline)) if a.baseline > 0 else calibrate_auto(m)
    summarize(m, thr)
    print('saved ->', plot(A, m, thr, a.out))

if __name__ == '__main__':
    main()