"""
Cross-Modal Supervision Loss for RF-Pose.
Measures keypoint confidence map consistency between Student RF model and Teacher visual model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalPoseLoss(nn.Module):
    """
    Cross-Modal Pose Supervision Loss.
    Calculates weighted L2 or Smooth L1 loss between RF Student keypoint heatmaps
    and Visual Teacher keypoint heatmaps.
    """
    def __init__(self, loss_type: str = "l2", keypoint_weights: torch.Tensor = None):
        super().__init__()
        assert loss_type in ["l2", "smooth_l1"], "loss_type must be 'l2' or 'smooth_l1'"
        self.loss_type = loss_type
        self.keypoint_weights = keypoint_weights

    def forward(
        self,
        student_heatmaps: torch.Tensor,
        teacher_heatmaps: torch.Tensor,
        visibility_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            student_heatmaps: (B, K, T, H, W) Output from RFPoseStudent
            teacher_heatmaps: (B, K, T, H, W) Output from RFPoseTeacher
            visibility_mask: Optional tensor of shape (B, K, T, 1, 1) or (B, K, T, H, W) with binary/float weights
            
        Returns:
            loss: Scalar tensor representing mean cross-modal supervision loss
        """
        # Ensure identical shapes
        if student_heatmaps.shape != teacher_heatmaps.shape:
            teacher_heatmaps = F.interpolate(
                teacher_heatmaps,
                size=student_heatmaps.shape[2:],
                mode='trilinear',
                align_corners=False
            )

        if self.loss_type == "l2":
            diff = (student_heatmaps - teacher_heatmaps) ** 2
        else:  # smooth_l1
            diff = F.smooth_l1_loss(student_heatmaps, teacher_heatmaps, reduction='none')

        # Apply spatial / keypoint visibility mask if provided
        if visibility_mask is not None:
            diff = diff * visibility_mask

        # Apply per-keypoint weighting if specified
        if self.keypoint_weights is not None:
            weights = self.keypoint_weights.to(diff.device).view(1, -1, 1, 1, 1)
            diff = diff * weights

        return diff.mean()
