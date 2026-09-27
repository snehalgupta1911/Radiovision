"""
Visual Teacher Network Interface (RFPoseTeacher)
Extracts keypoint confidence heatmaps from synchronized RGB camera frames for Cross-Modal Supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualKeypointBackbone(nn.Module):
    """
    2D CNN Pose Backbone operating on individual RGB frames.
    Maps (B * T, 3, H_img, W_img) -> (B * T, num_keypoints, H_out, W_out).
    """
    def __init__(self, num_keypoints: int = 18, base_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Conv2d(base_channels, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.deconv(x)
        x = self.head(x)
        return torch.sigmoid(x)  # Normalized confidence heatmaps [0, 1]


class RFPoseTeacher(nn.Module):
    """
    Visual Teacher Network for RF-Pose.
    Converts synchronized RGB video frames (B, 3, T, H_img, W_img) into target keypoint confidence maps.
    """
    def __init__(self, num_keypoints: int = 18, out_spatial_size: tuple = (48, 48)):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.out_spatial_size = out_spatial_size
        self.backbone = VisualKeypointBackbone(num_keypoints=num_keypoints)

    def forward(self, rgb_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_frames: Tensor of shape (B, 3, T, H_img, W_img)
            
        Returns:
            visual_confidence_maps: Tensor of shape (B, num_keypoints, T, H_out, W_out)
        """
        B, C, T, H_img, W_img = rgb_frames.shape

        # Reshape sequence: (B, C, T, H, W) -> (B * T, C, H, W)
        frames_flat = rgb_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H_img, W_img)

        # Forward pass through 2D pose estimator
        heatmaps_flat = self.backbone(frames_flat)  # (B * T, K, H_backbone, W_backbone)

        # Resize spatial size if needed
        _, K, H_curr, W_curr = heatmaps_flat.shape
        if (H_curr, W_curr) != self.out_spatial_size:
            heatmaps_flat = F.interpolate(
                heatmaps_flat,
                size=self.out_spatial_size,
                mode='bilinear',
                align_corners=False
            )

        # Reshape back to sequence: (B * T, K, H_out, W_out) -> (B, K, T, H_out, W_out)
        visual_confidence_maps = heatmaps_flat.reshape(B, T, K, self.out_spatial_size[0], self.out_spatial_size[1]).permute(0, 2, 1, 3, 4)

        return visual_confidence_maps
