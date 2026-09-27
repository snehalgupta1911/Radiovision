"""
Synthetic Data Generation Utilities for RF-Pose and WiFi CSI Testing.
"""

import torch


def generate_synthetic_rf_batch(
    batch_size: int = 2,
    temporal_frames: int = 30,
    spatial_size: tuple = (60, 60),
    num_keypoints: int = 18,
    rgb_size: tuple = (128, 128)
) -> dict:
    """
    Generates dummy batch of Vertical RF Heatmaps, Horizontal RF Heatmaps, and RGB frames.
    
    Returns:
        dict containing:
            - 'x_vert': (B, 1, T, H, W)
            - 'x_horiz': (B, 1, T, H, W)
            - 'rgb_frames': (B, 3, T, H_img, W_img)
            - 'visibility_mask': (B, K, T, 1, 1)
    """
    H, W = spatial_size
    H_img, W_img = rgb_size

    x_vert = torch.randn(batch_size, 1, temporal_frames, H, W)
    x_horiz = torch.randn(batch_size, 1, temporal_frames, H, W)
    rgb_frames = torch.rand(batch_size, 3, temporal_frames, H_img, W_img)
    visibility_mask = (torch.rand(batch_size, num_keypoints, temporal_frames, 1, 1) > 0.1).float()

    return {
        "x_vert": x_vert,
        "x_horiz": x_horiz,
        "rgb_frames": rgb_frames,
        "visibility_mask": visibility_mask
    }


def generate_synthetic_csi_batch(
    batch_size: int = 4,
    in_subcarriers: int = 90,
    temporal_frames: int = 100,
    num_gestures: int = 10,
    num_keypoints: int = 18
) -> dict:
    """
    Generates dummy batch of Channel State Information (CSI) matrices and ground truth targets.
    
    Returns:
        dict containing:
            - 'csi_data': (B, in_subcarriers, T)
            - 'gesture_targets': (B,) class labels [0, num_gestures-1]
            - 'coarse_pose_targets': (B, K, 2) keypoint coordinates
            - 'trajectory_targets': (B, T, 3) 3D spatial points
    """
    csi_data = torch.randn(batch_size, in_subcarriers, temporal_frames)
    gesture_targets = torch.randint(0, num_gestures, (batch_size,))
    coarse_pose_targets = torch.rand(batch_size, num_keypoints, 2)
    trajectory_targets = torch.randn(batch_size, temporal_frames, 3)

    return {
        "csi_data": csi_data,
        "gesture_targets": gesture_targets,
        "coarse_pose_targets": coarse_pose_targets,
        "trajectory_targets": trajectory_targets
    }
