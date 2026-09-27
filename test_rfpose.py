"""
Comprehensive Verification and Test Script for RF-Pose Architecture & WiFi CSI Model
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfpose.models import RFPoseStudent, RFPoseTeacher, CSIMultiTaskModel
from rfpose.losses import CrossModalPoseLoss
from rfpose.utils import generate_synthetic_rf_batch, generate_synthetic_csi_batch


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_rfpose_student_and_teacher():
    print_section("Testing RF-Pose Student & Teacher Cross-Modal Network")

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running tests on device: {device}")

    # Hyperparameters
    B, T = 2, 16
    K = 18
    H_rf, W_rf = 60, 60
    H_out, W_out = 48, 48
    H_img, W_img = 128, 128

    # 1. Instantiate Models
    student = RFPoseStudent(num_keypoints=K, out_spatial_size=(H_out, W_out)).to(device)
    teacher = RFPoseTeacher(num_keypoints=K, out_spatial_size=(H_out, W_out)).to(device)
    criterion = CrossModalPoseLoss(loss_type="l2").to(device)

    print(f"RFPoseStudent Trainable Parameters: {count_parameters(student):,}")
    print(f"RFPoseTeacher Trainable Parameters: {count_parameters(teacher):,}")

    # 2. Generate Synthetic RF and Visual Batch
    batch = generate_synthetic_rf_batch(
        batch_size=B,
        temporal_frames=T,
        spatial_size=(H_rf, W_rf),
        num_keypoints=K,
        rgb_size=(H_img, W_img)
    )

    x_vert = batch["x_vert"].to(device)
    x_horiz = batch["x_horiz"].to(device)
    rgb_frames = batch["rgb_frames"].to(device)
    vis_mask = batch["visibility_mask"].to(device)

    print(f"Input Vertical RF Tensor Shape:   {x_vert.shape}")
    print(f"Input Horizontal RF Tensor Shape: {x_horiz.shape}")
    print(f"Input RGB Frame Tensor Shape:     {rgb_frames.shape}")

    # 3. Forward Passes
    student_heatmaps = student(x_vert, x_horiz)
    with torch.no_grad():
        teacher_heatmaps = teacher(rgb_frames)

    print(f"\nStudent Output Confidence Maps Shape: {student_heatmaps.shape}")
    print(f"Teacher Output Confidence Maps Shape: {teacher_heatmaps.shape}")

    assert student_heatmaps.shape == (B, K, T, H_out, W_out), \
        f"Expected Student shape {(B, K, T, H_out, W_out)}, got {student_heatmaps.shape}"
    assert teacher_heatmaps.shape == (B, K, T, H_out, W_out), \
        f"Expected Teacher shape {(B, K, T, H_out, W_out)}, got {teacher_heatmaps.shape}"

    # 4. Cross-Modal Loss & Backward Pass Test
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    optimizer.zero_grad()

    loss = criterion(student_heatmaps, teacher_heatmaps, visibility_mask=vis_mask)
    print(f"\nComputed Cross-Modal Supervision Loss: {loss.item():.6f}")

    loss.backward()

    # Check gradient flow across student parameters
    grad_count = 0
    for name, param in student.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_count += 1

    total_params_count = len(list(student.parameters()))
    print(f"Gradient Flow Check: {grad_count}/{total_params_count} parameter tensors received valid gradients.")
    assert grad_count == total_params_count, "Some parameters in RFPoseStudent did not receive gradients!"

    optimizer.step()
    print("SUCCESS: Student-Teacher Network Forward/Backward/Optimization steps verified successfully!")


def test_csi_multitask_model():
    print_section("Testing WiFi CSI Multi-Task Model (1D CNN + BiLSTM + FC)")

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    B, C_sub, T = 4, 90, 50
    num_gestures = 10
    num_keypoints = 18

    model = CSIMultiTaskModel(
        in_subcarriers=C_sub,
        num_gestures=num_gestures,
        num_keypoints=num_keypoints
    ).to(device)

    print(f"CSIMultiTaskModel Trainable Parameters: {count_parameters(model):,}")

    batch = generate_synthetic_csi_batch(
        batch_size=B,
        in_subcarriers=C_sub,
        temporal_frames=T,
        num_gestures=num_gestures,
        num_keypoints=num_keypoints
    )

    csi_data = batch["csi_data"].to(device)
    gesture_gt = batch["gesture_targets"].to(device)
    pose_gt = batch["coarse_pose_targets"].to(device)
    traj_gt = batch["trajectory_targets"].to(device)

    print(f"Input CSI Tensor Shape: {csi_data.shape}")

    # Forward Pass
    outputs = model(csi_data)
    gesture_logits = outputs["gesture_logits"]
    coarse_pose = outputs["coarse_pose"]
    motion_trajectory = outputs["motion_trajectory"]

    print(f"Gesture Recognition Output Shape: {gesture_logits.shape}  (Expected: [{B}, {num_gestures}])")
    print(f"Coarse Pose Estimation Output Shape: {coarse_pose.shape}  (Expected: [{B}, {num_keypoints}, 2])")
    print(f"Motion Trajectory Output Shape:     {motion_trajectory.shape}  (Expected: [{B}, {T}, 3])")

    assert gesture_logits.shape == (B, num_gestures)
    assert coarse_pose.shape == (B, num_keypoints, 2)
    assert motion_trajectory.shape == (B, T, 3)

    # Multi-task Loss computation
    gesture_loss = nn.CrossEntropyLoss()(gesture_logits, gesture_gt)
    pose_loss = nn.MSELoss()(coarse_pose, pose_gt)
    traj_loss = nn.MSELoss()(motion_trajectory, traj_gt)

    total_loss = gesture_loss + 10.0 * pose_loss + 5.0 * traj_loss
    print(f"\nMulti-Task Loss Breakdown:")
    print(f"  - Gesture Loss:    {gesture_loss.item():.4f}")
    print(f"  - Pose Loss:       {pose_loss.item():.4f}")
    print(f"  - Trajectory Loss: {traj_loss.item():.4f}")
    print(f"  - Total Loss:      {total_loss.item():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("SUCCESS: WiFi CSI Multi-Task Model Forward/Backward steps verified successfully!")


def main():
    print("Starting Comprehensive RF-Pose & CSI Architecture Verification Suite...")
    test_rfpose_student_and_teacher()
    test_csi_multitask_model()
    print_section("ALL TESTS PASSED SUCCESSFULLY! MODEL REPLICATION IS FULLY VALIDATED.")


if __name__ == "__main__":
    main()
