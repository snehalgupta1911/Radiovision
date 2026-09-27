"""
ESP32 Wi-Fi CSI Raw Data Parser.
Supports parsing format from `csi_logger.py`, `csi_visualization.py`, PDF raw logs, and `.npy` recordings.
Handles `CSI,packet_idx,timestamp,rssi,len,[i0 q0 i1 q1 ...]` and `CSI_DATA, frame, rssi, noise, len, "I Q", "I Q", ...`.
"""

import numpy as np
import os
import re
from typing import List, Tuple, Union


class ESP32S3CSIParser:
    """
    Parser for ESP32 / XIAO ESP32-S3 Wi-Fi Channel State Information (CSI).
    Supports space-separated I/Q pairs, comma-separated I/Q, bracketed strings, and NumPy arrays.
    """
    def __init__(self, num_subcarriers: int = None):
        self.num_subcarriers = num_subcarriers

    def parse_line(self, line: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Parses a single serial log line from ESP32 CSI logger or PDF logs.
        Supports both:
          1) csi_logger.py format: CSI_DATA,frame,rssi,noise,len, "I0 Q0", "I1 Q1", ...
          2) csi_visualization.py / PDF format: CSI,packet_idx,timestamp,rssi,len,[i0 q0 i1 q1 ...]
          
        Returns:
            amp: 1D float32 array of shape (n_subcarriers,)
            iq: 2D int16 array of shape (n_subcarriers, 2)
            rssi: int RSSI value
        """
        line = line.strip()
        if not line or not (line.startswith("CSI_DATA,") or line.startswith("CSI,")):
            return None, None, None

        # Format 1: Bracketed format CSI,packet_idx,timestamp,rssi,len,[i0 q0 i1 q1 ...]
        if "[" in line and "]" in line:
            header, values = line.split("[", 1)
            values = values.split("]", 1)[0]
            header_parts = [p.strip() for p in header.split(",") if p.strip()]

            # Extract RSSI (usually index 3 in CSI,idx,timestamp,rssi,len format, or index 2 in CSI_DATA)
            rssi = 0
            for hp in header_parts:
                if hp.lstrip('-').isdigit():
                    val = int(hp)
                    if -110 <= val <= 0:  # Typical Wi-Fi RSSI range dBm
                        rssi = val
                        break

            numbers = [int(x) for x in re.findall(r"-?\d+", values)]
            if len(numbers) < 2:
                return None, None, None
            if len(numbers) % 2 != 0:
                numbers = numbers[:-1]

            real = np.array(numbers[0::2], dtype=np.float32)
            imag = np.array(numbers[1::2], dtype=np.float32)

            amp = np.sqrt(real**2 + imag**2)
            iq = np.column_stack((real, imag)).astype(np.int16)
            return amp, iq, rssi

        # Format 2: Space-separated I/Q pairs CSI_DATA,frame,rssi,noise,len, "I Q", "I Q", ...
        parts = line.split(",")
        if len(parts) < 6:
            return None, None, None

        try:
            rssi = int(parts[2])
        except ValueError:
            rssi = 0

        pair_tokens = parts[5:]
        iq_list = []
        for tok in pair_tokens:
            tok = tok.strip()
            if not tok:
                continue
            nums = tok.split()
            if len(nums) == 2:
                try:
                    i_val, q_val = int(nums[0]), int(nums[1])
                    iq_list.append((i_val, q_val))
                except ValueError:
                    continue
            elif len(nums) == 1 and tok.lstrip('-').isdigit():
                iq_list.append(int(nums[0]))

        if len(iq_list) == 0:
            return None, None, None

        if isinstance(iq_list[0], tuple):
            iq = np.array(iq_list, dtype=np.int16)
            amp = np.sqrt(iq[:, 0].astype(np.float32)**2 + iq[:, 1].astype(np.float32)**2)
            return amp, iq, rssi
        else:
            nums = iq_list
            if len(nums) % 2 != 0:
                nums = nums[:-1]
            real = np.array(nums[0::2], dtype=np.float32)
            imag = np.array(nums[1::2], dtype=np.float32)
            amp = np.sqrt(real**2 + imag**2)
            iq = np.column_stack((real, imag)).astype(np.int16)
            return amp, iq, rssi

    def parse_csv_stream(self, csv_lines: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parses a sequence of CSI log lines.
        
        Returns:
            amplitudes: 2D array of shape (T, n_subcarriers)
            raw_iq: 3D array of shape (T, n_subcarriers, 2)
            rssi_array: 1D array of shape (T,)
        """
        amp_list = []
        iq_list = []
        rssi_list = []

        for line in csv_lines:
            amp, iq, rssi = self.parse_line(line)
            if amp is not None:
                amp_list.append(amp)
                iq_list.append(iq)
                rssi_list.append(rssi)

        if not amp_list:
            n_sub = self.num_subcarriers or 64
            return np.zeros((1, n_sub), dtype=np.float32), np.zeros((1, n_sub, 2), dtype=np.int16), np.zeros((1,), dtype=np.int64)

        # Truncate to minimum subcarrier count across frames if variable
        min_sub = min(a.shape[0] for a in amp_list)
        amp_arr = np.array([a[:min_sub] for a in amp_list], dtype=np.float32)
        iq_arr = np.array([q[:min_sub] for q in iq_list], dtype=np.int16)
        rssi_arr = np.array(rssi_list, dtype=np.int64)

        return amp_arr, iq_arr, rssi_arr

    def load_npy_recording(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads saved NumPy recordings generated by `csi_logger.py` (data/<label>/<label>_<idx>.npy).
        
        Returns:
            amplitude: (n_frames, n_sub)
            phase: (n_frames, n_sub) computed if raw I/Q file exists, else phase is zeros.
        """
        amp = np.load(file_path).astype(np.float32)
        phase = np.zeros_like(amp)

        raw_path = file_path.replace(".npy", "_raw.npy")
        if os.path.exists(raw_path):
            raw_iq = np.load(raw_path).astype(np.float32)
            real = raw_iq[:, :, 0]
            imag = raw_iq[:, :, 1]
            phase = np.arctan2(imag, real)

        return amp, phase
