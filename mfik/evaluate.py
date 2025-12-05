import argparse
import os
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import matplotlib.pyplot as plt

from mfik.model import ConditionalMLP
from mfik.urdf import URDF


class IKDataset(Dataset):
    def __init__(self, data_path, split="test"):
        # Load PyTorch format dataset (GPU-friendly)
        data = torch.load(data_path, map_location="cpu")
        self.poses = data[f"poses_{split}"]
        self.joints = data[f"joints_{split}"]
        self.metadata = data["metadata"]

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return self.joints[idx], self.poses[idx]


@torch.no_grad()
def meanflow_ik_sampler(model, pose_target, q_ref=None, num_steps=1, device="cpu"):
    """
    MeanFlow IK sampler with Neighborhood Projection.
    
    Two inference modes (from method.md):
    1. Trajectory tracking: Given current q_ref, find nearest solution for pose_target
    2. Global solving: Use random q_ref to discover all possible solutions
    
    Args:
        model: ConditionalMLP model
        pose_target: (B, 7) Target end-effector pose
        q_ref: (B, n_joints) Reference joint configuration. If None, use random.
        num_steps: Number of sampling steps
        device: Device
    Returns:
        q_pred: (B, n_joints) Predicted joint configuration (nearest to q_ref)
    """
    batch_size = pose_target.shape[0]
    n_joints = model.n_joints

    # If no q_ref provided, use random initialization (global solving mode)
    if q_ref is None:
        q_ref = torch.randn(batch_size, n_joints, device=device)
    
    # Start from q_ref (this is our t=0 state)
    z = q_ref.clone()
    
    # Noise scale: set to 0 for inference (we want to project to the exact manifold)
    c = torch.zeros(batch_size, 1, device=device)

    if num_steps == 1:
        # Single-step: Flow from t=0 (q_ref) to t=1 (q_gt)
        # q_pred = z_0 + u(z_0, 0, 1 | pose_target, c=0)
        # Since u predicts velocity v = q_gt - q_ref, we add it
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        u = model(z, pose_target, r, t, c)
        q_pred = z + u  # Move from q_ref towards the solution
    else:
        # Multi-step: iterate from t=0 to t=1
        time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]

            t = torch.full((batch_size,), t_cur, device=device)
            r = torch.full((batch_size,), t_next, device=device)

            u = model(z, pose_target, t, r, c)

            # Update: z_next = z_cur + (t_next - t_cur) * u
            z = z + (t_next - t_cur) * u

        q_pred = z

    return q_pred


@torch.no_grad()
def meanflow_ik_global_solver(model, pose_target, num_random_starts=16, num_steps=1, device="cpu"):
    """
    Global IK solver that discovers multiple solutions by using diverse q_ref.
    
    This implements the "Global solving" mode from method.md:
    - Generate K random q_ref points
    - Each q_ref gets "attracted" to its nearest valid IK solution
    - Cluster results to find unique solutions
    
    Args:
        model: ConditionalMLP model
        pose_target: (1, 7) Single target pose
        num_random_starts: Number of random starting points
        num_steps: Number of flow steps per solution
        device: Device
    Returns:
        solutions: List of unique joint configurations
    """
    n_joints = model.n_joints
    
    # Expand pose_target to match num_random_starts
    pose_batch = pose_target.expand(num_random_starts, -1)
    
    # Generate diverse random starting points
    q_refs = torch.randn(num_random_starts, n_joints, device=device) * 2.0  # Larger spread
    
    # Run flow from each q_ref
    q_preds = meanflow_ik_sampler(model, pose_batch, q_ref=q_refs, num_steps=num_steps, device=device)
    
    # Cluster to find unique solutions (simple distance-based)
    solutions = []
    q_preds_np = q_preds.cpu().numpy()
    
    for q in q_preds_np:
        is_new = True
        for existing in solutions:
            if np.linalg.norm(q - existing) < 0.1:  # Threshold for "same" solution
                is_new = False
                break
        if is_new:
            solutions.append(q)
    
    return solutions


def quaternion_distance(q1, q2):
    """
    Compute angular distance between two quaternions (in degrees)
    Args:
        q1, q2: (..., 4) quaternions (x, y, z, w)
    Returns:
        angle: angular distance in degrees
    """
    # Normalize quaternions
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-8)

    # Compute dot product
    dot = np.clip(np.sum(q1 * q2, axis=-1), -1.0, 1.0)

    # Angular distance
    angle_rad = 2 * np.arccos(np.abs(dot))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def evaluate_model(model, data_loader, robot, end_effector, num_steps=1, device="cpu", use_gt_ref=False):
    """
    Evaluate IK model
    
    Args:
        use_gt_ref: If True, use ground truth joints as q_ref (trajectory tracking mode)
                    If False, use random q_ref (global solving mode)
    """
    model.eval()

    all_pos_errors = []
    all_rot_errors = []
    all_joint_errors = []
    inference_times = []

    for joints_true, pose_target in tqdm(data_loader, desc="Evaluating"):
        pose_target = pose_target.to(device)
        joints_true_tensor = joints_true.to(device)
        joints_true = joints_true.numpy()

        batch_size = pose_target.shape[0]

        # Set q_ref based on evaluation mode
        if use_gt_ref:
            # Trajectory tracking mode: use GT with small noise
            q_ref = joints_true_tensor + torch.randn_like(joints_true_tensor) * 0.1
        else:
            # Global solving mode: use random q_ref
            q_ref = None

        # Measure inference time
        start_time = time.time()
        joints_pred = meanflow_ik_sampler(
            model, pose_target, q_ref=q_ref, num_steps=num_steps, device=device
        )
        inference_times.append((time.time() - start_time) / batch_size)

        joints_pred = joints_pred.cpu().numpy()

        # Compute FK for predicted joints
        # Convert to list of configs for FK
        joints_pred_list = [joints_pred[i] for i in range(batch_size)]
        transforms_pred = robot.link_fk_batch(cfgs=joints_pred_list, link=end_effector)

        # Extract position and quaternion from predicted transforms
        positions_pred = transforms_pred[:, :3, 3]
        rot_matrices_pred = transforms_pred[:, :3, :3]
        quaternions_pred = Rotation.from_matrix(rot_matrices_pred).as_quat()

        # Extract from target pose (pose_target is already 7D: pos + quat)
        positions_target = pose_target[:, :3].cpu().numpy()
        quaternions_target = pose_target[:, 3:].cpu().numpy()

        # Compute errors
        pos_errors = np.linalg.norm(positions_pred - positions_target, axis=1)
        rot_errors = quaternion_distance(quaternions_pred, quaternions_target)
        joint_errors = np.linalg.norm(joints_pred - joints_true, axis=1)

        all_pos_errors.extend(pos_errors.tolist())
        all_rot_errors.extend(rot_errors.tolist())
        all_joint_errors.extend(joint_errors.tolist())

    all_pos_errors = np.array(all_pos_errors)
    all_rot_errors = np.array(all_rot_errors)
    all_joint_errors = np.array(all_joint_errors)

    # Compute success rate (position < 1cm and rotation < 5°)
    success_mask = (all_pos_errors < 0.01) & (all_rot_errors < 5.0)
    success_rate = success_mask.mean() * 100

    # Compute statistics
    metrics = {
        "success_rate": success_rate,
        "position_error": {
            "mean": float(all_pos_errors.mean()),
            "std": float(all_pos_errors.std()),
            "median": float(np.median(all_pos_errors)),
            "p95": float(np.percentile(all_pos_errors, 95)),
        },
        "rotation_error": {
            "mean": float(all_rot_errors.mean()),
            "std": float(all_rot_errors.std()),
            "median": float(np.median(all_rot_errors)),
            "p95": float(np.percentile(all_rot_errors, 95)),
        },
        "joint_error": {
            "mean": float(all_joint_errors.mean()),
            "std": float(all_joint_errors.std()),
            "median": float(np.median(all_joint_errors)),
        },
        "inference_speed": {
            "mean_time_per_sample": float(np.mean(inference_times)),
            "std_time_per_sample": float(np.std(inference_times)),
            "samples_per_sec": float(1.0 / np.mean(inference_times)),
        },
    }

    return metrics, all_pos_errors, all_rot_errors, all_joint_errors


def plot_error_distributions(pos_errors, rot_errors, joint_errors, output_path):
    """Plot error distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(pos_errors * 1000, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(10, color="r", linestyle="--", label="Success threshold (10mm)")
    axes[0].set_xlabel("Position Error (mm)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Position Error Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(rot_errors, bins=50, edgecolor="black", alpha=0.7)
    axes[1].axvline(5, color="r", linestyle="--", label="Success threshold (5°)")
    axes[1].set_xlabel("Rotation Error (degrees)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Rotation Error Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(joint_errors, bins=50, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("Joint Error (radians)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Joint Space Error Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved error distribution plots to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to dataset .pt")
    parser.add_argument("--urdf", type=str, required=True, help="Path to URDF file")
    parser.add_argument(
        "--end-effector", type=str, required=True, help="End effector link name"
    )
    parser.add_argument(
        "--num-steps", type=int, default=1, help="Number of sampling steps"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--use-gt-ref", action="store_true",
        help="Use ground truth joints as q_ref (trajectory tracking mode)"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Infer n_joints from checkpoint or data
    data = torch.load(args.data, map_location="cpu")
    n_joints = data["joints_test"].shape[1]

    model = ConditionalMLP(n_joints=n_joints).to(args.device)

    # Try to load EMA model first, fallback to regular model
    if "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
        print("Loaded EMA model")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded regular model")

    # Load robot
    print(f"Loading URDF from {args.urdf}...")
    robot = URDF.load(args.urdf)

    # Load test data
    print(f"Loading test data from {args.data}...")
    test_dataset = IKDataset(args.data, split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    mode_str = "trajectory tracking" if args.use_gt_ref else "global solving"
    print(f"Evaluating with {args.num_steps}-step sampling ({mode_str} mode)...")
    metrics, pos_errors, rot_errors, joint_errors = evaluate_model(
        model,
        test_loader,
        robot,
        args.end_effector,
        num_steps=args.num_steps,
        device=args.device,
        use_gt_ref=args.use_gt_ref,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Success Rate: {metrics['success_rate']:.2f}%")
    print(f"\nPosition Error (mm):")
    print(f"  Mean: {metrics['position_error']['mean']*1000:.2f}")
    print(f"  Std:  {metrics['position_error']['std']*1000:.2f}")
    print(f"  Median: {metrics['position_error']['median']*1000:.2f}")
    print(f"  P95: {metrics['position_error']['p95']*1000:.2f}")
    print(f"\nRotation Error (degrees):")
    print(f"  Mean: {metrics['rotation_error']['mean']:.2f}")
    print(f"  Std:  {metrics['rotation_error']['std']:.2f}")
    print(f"  Median: {metrics['rotation_error']['median']:.2f}")
    print(f"  P95: {metrics['rotation_error']['p95']:.2f}")
    print(f"\nJoint Error (radians):")
    print(f"  Mean: {metrics['joint_error']['mean']:.4f}")
    print(f"  Std:  {metrics['joint_error']['std']:.4f}")
    print(f"\nInference Speed:")
    print(
        f"  Mean time per sample: {metrics['inference_speed']['mean_time_per_sample']*1000:.2f} ms"
    )
    print(
        f"  Throughput: {metrics['inference_speed']['samples_per_sec']:.1f} samples/sec"
    )
    print("=" * 60)

    # Save report
    report = {
        "config": vars(args),
        "metrics": metrics,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to {args.output}")

    # Plot distributions
    plot_path = args.output.replace(".json", "_plots.png")
    plot_error_distributions(pos_errors, rot_errors, joint_errors, plot_path)


if __name__ == "__main__":
    main()
