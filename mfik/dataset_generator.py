import argparse
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mfik.urdf import URDF
import tqdm


def generate_dataset(urdf_path, num_samples, output_path, end_effector, seed=42):
    np.random.seed(seed)

    # Load URDF
    print(f"Loading URDF from {urdf_path}")
    robot = URDF.load(urdf_path)

    # Get actuated joints and limits
    joints = robot.actuated_joints
    n_joints = len(joints)
    print(f"Found {n_joints} actuated joints: {[j.name for j in joints]}")

    lb, ub = robot.joint_limit_cfgs
    lower_limits = np.array([lb.get(j, -np.pi) for j in joints])
    upper_limits = np.array([ub.get(j, np.pi) for j in joints])

    # Handle unlimited joints or very large limits (optional, here we assume valid limits or clip)
    # For now, just use what's in URDF. If None, default to -pi/pi is maybe too small for continuous joints,
    # but for robotic arms usually limits are defined.

    print("Generating samples...")
    # Sample joint configurations
    # Shape: (num_samples, n_joints)
    q_samples = np.random.uniform(
        low=lower_limits, high=upper_limits, size=(num_samples, n_joints)
    )

    # Compute FK in batches to avoid OOM
    batch_size = 1000
    poses_list = []

    print(f"Computing FK for {num_samples} samples...")
    for i in tqdm.tqdm(range(0, num_samples, batch_size)):
        batch_q = q_samples[i : i + batch_size]

        # link_fk_batch returns {link: transform} or transform if link specified
        # We specify link=end_effector
        transforms = robot.link_fk_batch(cfgs=batch_q, link=end_effector)

        # transforms: (batch, 4, 4)
        poses_list.append(transforms)

    all_transforms = np.concatenate(poses_list, axis=0)

    # Extract position and quaternion
    # Position: (N, 3)
    positions = all_transforms[:, :3, 3]

    # Rotation: (N, 3, 3) -> Quaternion (N, 4)
    rot_matrices = all_transforms[:, :3, :3]
    rotations = Rotation.from_matrix(rot_matrices).as_quat()  # (x, y, z, w)

    # Concatenate to (N, 7) -> (px, py, pz, qx, qy, qz, qw)
    poses_7d = np.concatenate([positions, rotations], axis=1)

    # Split into train/val/test (80/10/10)
    n_train = int(num_samples * 0.8)
    n_val = int(num_samples * 0.1)
    n_test = num_samples - n_train - n_val

    # Convert to PyTorch tensors for GPU-friendly format
    data = {
        "poses_train": torch.from_numpy(poses_7d[:n_train]).float(),
        "joints_train": torch.from_numpy(q_samples[:n_train]).float(),
        "poses_val": torch.from_numpy(poses_7d[n_train : n_train + n_val]).float(),
        "joints_val": torch.from_numpy(q_samples[n_train : n_train + n_val]).float(),
        "poses_test": torch.from_numpy(poses_7d[n_train + n_val :]).float(),
        "joints_test": torch.from_numpy(q_samples[n_train + n_val :]).float(),
        "metadata": {
            "robot_name": robot.name,
            "joint_names": [j.name for j in joints],
            "lower_limits": torch.from_numpy(lower_limits).float(),
            "upper_limits": torch.from_numpy(upper_limits).float(),
            "end_effector": end_effector,
        },
    }

    # Save as PyTorch format for GPU-friendly loading
    print(f"Saving dataset to {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(data, output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, required=True, help="Path to URDF file")
    parser.add_argument(
        "--num-samples", type=int, default=100000, help="Number of samples"
    )
    parser.add_argument("--output", type=str, required=True, help="Output path (.pt)")
    parser.add_argument(
        "--end-effector", type=str, required=True, help="End effector link name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_dataset(
        args.urdf, args.num_samples, args.output, args.end_effector, args.seed
    )
