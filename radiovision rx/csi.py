"""
CSI ingestion for Radiovision.

Parses the serial lines emitted by `radiovision rx` firmware:

    CSI,<timestamp>,<rssi>,<len>,[b0 b1 b2 ... b(len-1)]

The ESP32 CSI `buf` is a sequence of signed int8 values laid out as
interleaved (imag, real) pairs, one pair per OFDM subcarrier. So `len`
bytes decode to `len // 2` complex subcarriers. This module turns a raw
serial line into a complex CSI vector plus amplitude / (sanitized) phase.

This is the one part of the pipeline that is fixed regardless of which
student architecture you end up choosing, so it's fully implemented here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class CSIFrame:
    timestamp: int          # ESP32 rx_ctrl.timestamp (microseconds, wraps)
    rssi: int               # dBm
    csi: np.ndarray         # complex64, shape (n_subcarriers,)
    host_time: float | None = None  # wall-clock time.time() when host read the line

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.csi)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.csi)


def parse_line(line: str, host_time: float | None = None) -> CSIFrame | None:
    """Parse one 'CSI,...' serial line into a CSIFrame, or None if malformed."""
    line = line.strip()
    if not line.startswith("CSI,"):
        return None
    try:
        head, rest = line.split(",[", 1)
        _tag, ts, rssi, ln = head.split(",")
        body = rest.rstrip("]").strip()
        raw = np.fromstring(body, dtype=np.int8, sep=" ") if body else np.empty(0, np.int8)
    except Exception:
        return None
    if raw.size < 2 or raw.size % 2 != 0:
        return None
    imag = raw[0::2].astype(np.float32)
    real = raw[1::2].astype(np.float32)
    csi = (real + 1j * imag).astype(np.complex64)
    return CSIFrame(timestamp=int(ts), rssi=int(rssi), csi=csi, host_time=host_time)


def sanitize_phase(phase: np.ndarray) -> np.ndarray:
    """
    Remove the linear phase slope across subcarriers caused by CFO/SFO and
    sample-timing offset (the classic ESP32/Intel-5300 CSI phase problem).
    Fits a line vs subcarrier index and subtracts it. Amplitude is untouched
    and is generally the more reliable feature to start from.
    """
    n = phase.shape[-1]
    k = np.arange(n)
    unwrapped = np.unwrap(phase, axis=-1)
    # slope a and offset b of the least-squares line, then subtract a*k + b
    a = (unwrapped[..., -1] - unwrapped[..., 0]) / (n - 1)
    b = unwrapped.mean(axis=-1)
    return unwrapped - (np.outer(a, k) if unwrapped.ndim > 1 else a * k) - b[..., None]


def drop_null_subcarriers(csi: np.ndarray, keep: slice | np.ndarray | None = None) -> np.ndarray:
    """Null/pilot/guard subcarriers carry no useful channel info. Override
    `keep` once you've confirmed your board's active-subcarrier indices."""
    if keep is None:
        return csi
    return csi[..., keep]