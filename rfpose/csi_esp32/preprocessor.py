"""
CSI Preprocessor for XIAO ESP32-S3 Wi-Fi Sensing.
Performs linear phase sanitization, noise filtering (Butterworth / Gaussian Lowpass),
static background subtraction, Z-score normalization, and temporal sliding window segmentation.
"""

import numpy as np
import torch
from typing import Tuple, List

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class CSIPreprocessor:
    """
    Complete Preprocessing Pipeline for raw ESP32-S3 Wi-Fi CSI matrices.
    """
    def __init__(
        self,
        cutoff_freq: float = 10.0,
        fs: float = 50.0,
        filter_order: int = 4,
        window_size: int = 100,
        stride: int = 25
    ):
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.filter_order = filter_order
        self.window_size = window_size
        self.stride = stride

    def sanitize_phase(self, phase_matrix: np.ndarray) -> np.ndarray:
        """
        Linear phase sanitization across subcarrier indices.
        Unwraps phase and fits linear slope a * k + b to remove CFO/STO.
        
        Args:
            phase_matrix: 2D array of shape (T, num_subcarriers)
            
        Returns:
            sanitized_phase: 2D array of shape (T, num_subcarriers)
        """
        T, N_sub = phase_matrix.shape
        sanitized = np.zeros_like(phase_matrix)
        k_indices = np.arange(N_sub, dtype=np.float32)

        for t in range(T):
            unwrapped = np.unwrap(phase_matrix[t])
            # Linear regression: unwrapped ~ a * k + b
            A_mat = np.vstack([k_indices, np.ones(N_sub)]).T
            a, b = np.linalg.lstsq(A_mat, unwrapped, rcond=None)[0]
            sanitized[t] = unwrapped - (a * k_indices + b)

        return sanitized

    def butterworth_lowpass(self, amplitude_matrix: np.ndarray) -> np.ndarray:
        """
        Applies zero-phase Butterworth lowpass filter along time dimension T.
        Fallback to Gaussian/moving window smoothing if scipy is unavailable.
        """
        T, N_sub = amplitude_matrix.shape
        if T <= 15:
            return amplitude_matrix

        if HAS_SCIPY:
            nyquist = 0.5 * self.fs
            normal_cutoff = min(self.cutoff_freq / nyquist, 0.99)
            b, a = butter(self.filter_order, normal_cutoff, btype='low', analog=False)

            filtered = np.zeros_like(amplitude_matrix)
            for s in range(N_sub):
                filtered[:, s] = filtfilt(b, a, amplitude_matrix[:, s])
            return filtered
        else:
            # Pure NumPy Gaussian lowpass kernel smoothing fallback
            kernel_size = 5
            kernel = np.exp(-np.linspace(-2, 2, kernel_size)**2)
            kernel /= kernel.sum()

            filtered = np.zeros_like(amplitude_matrix)
            for s in range(N_sub):
                filtered[:, s] = np.convolve(amplitude_matrix[:, s], kernel, mode='same')
            return filtered

    def subtract_background(self, amplitude_matrix: np.ndarray, bg_window: int = 20) -> np.ndarray:
        """
        Subtracts moving average static background component to highlight dynamic human motion.
        
        Args:
            amplitude_matrix: 2D array of shape (T, num_subcarriers)
            bg_window: Moving average window size
            
        Returns:
            motion_matrix: 2D array of shape (T, num_subcarriers)
        """
        T, N_sub = amplitude_matrix.shape
        bg = np.zeros_like(amplitude_matrix)
        for t in range(T):
            start_idx = max(0, t - bg_window)
            bg[t] = np.mean(amplitude_matrix[start_idx : t + 1], axis=0)

        return amplitude_matrix - bg

    def normalize(self, data_matrix: np.ndarray) -> np.ndarray:
        """
        Z-score normalization (zero mean, unit variance) per subcarrier.
        """
        mean = np.mean(data_matrix, axis=0, keepdims=True)
        std = np.std(data_matrix, axis=0, keepdims=True) + 1e-8
        return (data_matrix - mean) / std

    def segment_sliding_windows(self, data_matrix: np.ndarray) -> np.ndarray:
        """
        Slices a continuous matrix (T, N_subcarriers) into sliding window batches.
        
        Returns:
            windows: 3D array of shape (num_windows, N_subcarriers, window_size)
        """
        T, N_sub = data_matrix.shape
        if T < self.window_size:
            # Pad sequence if shorter than window_size
            pad_len = self.window_size - T
            padded = np.pad(data_matrix, ((0, pad_len), (0, 0)), mode='edge')
            return np.expand_dims(padded.T, axis=0)  # Shape: (1, N_sub, window_size)

        windows = []
        for start in range(0, T - self.window_size + 1, self.stride):
            end = start + self.window_size
            segment = data_matrix[start:end, :].T  # Shape: (N_sub, window_size)
            windows.append(segment)

        return np.array(windows, dtype=np.float32)

    def process(self, amplitude: np.ndarray, phase: np.ndarray = None) -> torch.Tensor:
        """
        Full end-to-end preprocessing pipeline execution.
        
        Args:
            amplitude: (T, N_subcarriers)
            phase: Optional (T, N_subcarriers)
            
        Returns:
            tensor_windows: PyTorch Tensor of shape (num_windows, N_subcarriers, window_size)
        """
        # 1. Filter lowpass noise
        amp_filtered = self.butterworth_lowpass(amplitude)

        # 2. Subtract static background
        amp_motion = self.subtract_background(amp_filtered)

        # 3. Z-score normalization
        amp_norm = self.normalize(amp_motion)

        # 4. Temporal sliding window segmentation
        windows_np = self.segment_sliding_windows(amp_norm)

        # 5. Convert to PyTorch Tensor
        return torch.from_numpy(windows_np).float()
