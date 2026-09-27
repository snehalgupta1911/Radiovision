"""
ESP32 Wi-Fi CSI Gesture Recognition Network (ESP32S3GestureNet).
Architecture: 1D Spatial CNN -> Bidirectional LSTM (Bi-LSTM) -> Fully Connected ANN Classifier.
Configured for RadioVision target gestures: ['sitting', 'moving right', 'swipe up', 'raise hand'].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

DEFAULT_RADIOVISION_GESTURES = ["sitting", "moving right", "swipe up", "raise hand"]


class ESP32S3GestureNet(nn.Module):
    """
    Gesture Classification Model for ESP32 / XIAO ESP32-S3 Wi-Fi CSI.
    
    Inputs:
      - csi_tensor: (B, in_subcarriers, T) Preprocessed CSI amplitude/phase tensor
      
    Outputs:
      - gesture_logits: (B, num_gestures) Raw classification logits per sample
    """
    def __init__(
        self,
        in_subcarriers: int = 64,
        num_gestures: int = 4,
        cnn_channels: list = [64, 128, 256],
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        dropout_prob: float = 0.3,
        gesture_labels: List[str] = DEFAULT_RADIOVISION_GESTURES
    ):
        super().__init__()
        self.in_subcarriers = in_subcarriers
        self.num_gestures = len(gesture_labels) if gesture_labels else num_gestures
        self.gesture_labels = gesture_labels or DEFAULT_RADIOVISION_GESTURES

        # 1. 1D Spatial CNN Feature Extractor (Processes subcarrier spatial correlations per timestep)
        cnn_layers = []
        curr_in = in_subcarriers
        for out_c in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(curr_in, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1)
            ])
            curr_in = out_c
        self.spatial_cnn = nn.Sequential(*cnn_layers)

        # 2. Bidirectional LSTM (Bi-LSTM) Temporal Extractor (Models sequential temporal dynamics)
        self.bilstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if lstm_num_layers > 1 else 0.0
        )

        bilstm_out_dim = lstm_hidden_dim * 2  # Bidirectional (Forward + Backward)

        # 3. Fully Connected ANN Classifier Head
        self.fc_classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob / 2.0),
            nn.Linear(64, self.num_gestures)
        )

    def forward(self, csi_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            csi_tensor: Tensor of shape (B, in_subcarriers, T)
            
        Returns:
            gesture_logits: Tensor of shape (B, num_gestures)
        """
        B, C, T = csi_tensor.shape

        # 1D CNN Spatial Feature Extraction: (B, C, T) -> (B, cnn_channels[-1], T)
        cnn_feat = self.spatial_cnn(csi_tensor)

        # Prepare for Bi-LSTM: (B, C, T) -> (B, T, C)
        lstm_in = cnn_feat.permute(0, 2, 1)

        # Bi-LSTM Temporal Sequence Modeling
        lstm_out, _ = self.bilstm(lstm_in)  # lstm_out shape: (B, T, bilstm_out_dim)

        # Global Temporal Feature Aggregation (Average Pooling across T frames)
        global_temporal_feat = lstm_out.mean(dim=1)  # (B, bilstm_out_dim)

        # Fully Connected ANN Gesture Classification
        gesture_logits = self.fc_classifier(global_temporal_feat)

        return gesture_logits

    def predict_gesture(self, csi_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper method returning predicted class IDs and probability distributions.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(csi_tensor)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs
