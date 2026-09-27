"""
WiFi CSI Multi-Task Deep Learning Model (Figure 2 in Paper)
Combines 1D Spatial CNN, Bidirectional LSTM (Bi-LSTM), and Fully Connected ANN heads
for Gesture Recognition, Coarse Pose Estimation, and Motion Tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSIMultiTaskModel(nn.Module):
    """
    WiFi CSI Multi-Task Model Architecture.
    
    Inputs:
      - csi_data: (B, in_subcarriers, T) Channel State Information sequence over T time steps
      
    Outputs:
      - gesture_logits: (B, num_gestures) Classification logits for static & dynamic gesture recognition
      - coarse_pose: (B, num_keypoints, 2) Coarse 2D keypoint coordinates normalized to [0, 1]
      - motion_trajectory: (B, T, 3) 3D spatial trajectory (X, Y, Z) over time
    """
    def __init__(
        self,
        in_subcarriers: int = 90,
        cnn_channels: list = [64, 128, 256],
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        num_gestures: int = 10,
        num_keypoints: int = 18
    ):
        super().__init__()
        self.in_subcarriers = in_subcarriers
        self.num_gestures = num_gestures
        self.num_keypoints = num_keypoints

        # 1D Spatial CNN Feature Extractor (along subcarriers per timestep)
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
        self.cnn_extractor = nn.Sequential(*cnn_layers)

        # Bidirectional LSTM (Bi-LSTM) Temporal Extractor
        self.bilstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_num_layers > 1 else 0.0
        )

        bilstm_out_dim = lstm_hidden_dim * 2  # Bidirectional

        # Fully Connected ANN Multi-Task Heads
        # 1. Gesture Recognition Head (Global Temporal Representation)
        self.gesture_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_gestures)
        )

        # 2. Coarse Pose Estimation Head (Keypoint Coordinate Regression)
        self.coarse_pose_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_keypoints * 2),
            nn.Sigmoid()  # Coordinates normalized in range [0, 1]
        )

        # 3. Motion Tracking Head (Per-frame 3D Position Regression)
        self.motion_tracking_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # (X, Y, Z) coordinates per frame
        )

    def forward(self, csi_data: torch.Tensor) -> dict:
        """
        Args:
            csi_data: Tensor of shape (B, in_subcarriers, T)
            
        Returns:
            dict with keys:
              - 'gesture_logits': (B, num_gestures)
              - 'coarse_pose': (B, num_keypoints, 2)
              - 'motion_trajectory': (B, T, 3)
        """
        B, C, T = csi_data.shape

        # 1. 1D CNN Spatial Feature Extraction per timestamp
        cnn_feat = self.cnn_extractor(csi_data)  # Shape: (B, cnn_channels[-1], T)

        # 2. Prepare for Bi-LSTM: (B, C, T) -> (B, T, C)
        lstm_in = cnn_feat.permute(0, 2, 1)

        # 3. Bi-LSTM Temporal Modeling
        lstm_out, (h_n, c_n) = self.bilstm(lstm_in)  # lstm_out: (B, T, bilstm_out_dim)

        # Global temporal representation (Average Pooling across time sequence)
        global_temporal_feat = lstm_out.mean(dim=1)  # (B, bilstm_out_dim)

        # 4. Multi-Task ANN Predictions
        gesture_logits = self.gesture_head(global_temporal_feat)
        
        coarse_pose = self.coarse_pose_head(global_temporal_feat).view(B, self.num_keypoints, 2)
        
        motion_trajectory = self.motion_tracking_head(lstm_out)  # (B, T, 3)

        return {
            "gesture_logits": gesture_logits,
            "coarse_pose": coarse_pose,
            "motion_trajectory": motion_trajectory
        }
