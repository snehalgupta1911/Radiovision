"""
RF-Pose Student Network Architecture (Zhao et al., CVPR 2018)
Spatio-Temporal 3D-CNN Dual-Branch Encoder with Cross-Axis Feature Fusion
and Keypoint Confidence Map Decoder for Through-Wall Pose Estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalConvBlock3d(nn.Module):
    """
    3D Convolutional Block with BatchNorm and ReLU for Spatio-Temporal Feature Extraction.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=(1, 1, 1),
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class VerticalEncoder(nn.Module):
    """
    Vertical RF Heatmap Encoder.
    Extracts spatio-temporal features from vertical projection heatmaps X_v (B, 1, T, H_v, W_v).
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.layer1 = SpatioTemporalConvBlock3d(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Preserve time resolution initially

        self.layer2 = SpatioTemporalConvBlock3d(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.layer3 = SpatioTemporalConvBlock3d(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        return x


class HorizontalEncoder(nn.Module):
    """
    Horizontal RF Heatmap Encoder.
    Extracts spatio-temporal features from horizontal projection heatmaps X_h (B, 1, T, H_h, W_h).
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.layer1 = SpatioTemporalConvBlock3d(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.layer2 = SpatioTemporalConvBlock3d(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.layer3 = SpatioTemporalConvBlock3d(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        return x


class FeatureFusion(nn.Module):
    """
    Cross-Axis Feature Fusion Module.
    Combines spatio-temporal feature maps from Vertical and Horizontal Encoders.
    """
    def __init__(self, in_channels: int = 256, out_channels: int = 128):
        super().__init__()
        self.fusion_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, vert_feat: torch.Tensor, horiz_feat: torch.Tensor) -> torch.Tensor:
        # Match spatial resolution if needed before concatenation
        if vert_feat.shape[3:] != horiz_feat.shape[3:]:
            horiz_feat = F.interpolate(horiz_feat, size=vert_feat.shape[2:], mode='trilinear', align_corners=False)
        
        fused = torch.cat([vert_feat, horiz_feat], dim=1)  # Concatenate along channel dim
        fused = self.fusion_conv1(fused)
        fused = self.fusion_conv2(fused)
        return fused


class PoseDecoder(nn.Module):
    """
    Keypoint Confidence Map Decoder.
    Upsamples fused spatio-temporal representations into 2D keypoint confidence heatmaps over time.
    """
    def __init__(self, in_channels: int = 128, num_keypoints: int = 18, target_size: tuple = (48, 48)):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.target_size = target_size

        # Upsampling deconv / transposed conv stages
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv3d(32, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.final_conv(x)  # (B, num_keypoints, T_deconv, H_deconv, W_deconv)

        # Interpolate to match target temporal length T and spatial resolution (H_out, W_out)
        B, K, T_curr, H_curr, W_curr = x.shape
        if T_curr != target_frames or (H_curr, W_curr) != self.target_size:
            x = F.interpolate(
                x,
                size=(target_frames, self.target_size[0], self.target_size[1]),
                mode='trilinear',
                align_corners=False
            )
        
        return x  # Shape: (B, num_keypoints, T, H_out, W_out)


class RFPoseStudent(nn.Module):
    """
    Unified RF-Pose Student Network (Zhao et al., CVPR 2018).
    
    Inputs:
      - x_vert: (B, 1, T, H_v, W_v) Vertical RF heatmap sequence across time
      - x_horiz: (B, 1, T, H_h, W_h) Horizontal RF heatmap sequence across time
      
    Outputs:
      - confidence_maps: (B, num_keypoints, T, H_out, W_out) 2D Keypoint confidence maps over T frames
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_keypoints: int = 18,
        out_spatial_size: tuple = (48, 48)
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.out_spatial_size = out_spatial_size

        self.vert_encoder = VerticalEncoder(in_channels=in_channels, base_channels=base_channels)
        self.horiz_encoder = HorizontalEncoder(in_channels=in_channels, base_channels=base_channels)
        self.fusion = FeatureFusion(in_channels=base_channels * 4 * 2, out_channels=base_channels * 4)
        self.decoder = PoseDecoder(in_channels=base_channels * 4, num_keypoints=num_keypoints, target_size=out_spatial_size)

    def forward(self, x_vert: torch.Tensor, x_horiz: torch.Tensor) -> torch.Tensor:
        target_frames = x_vert.shape[2]
        
        # 1. Encode Vertical and Horizontal Spatio-Temporal RF Heatmaps
        vert_feat = self.vert_encoder(x_vert)
        horiz_feat = self.horiz_encoder(x_horiz)

        # 2. Fuse Cross-Axis Features
        fused_feat = self.fusion(vert_feat, horiz_feat)

        # 3. Decode into Keypoint Confidence Heatmaps
        keypoint_confidence_maps = self.decoder(fused_feat, target_frames=target_frames)

        return keypoint_confidence_maps
