"""
Dataset utilities for loading and generating FK datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import time
from pathlib import Path
from tqdm import tqdm


class FKDataset(Dataset):
    """
    PyTorch Dataset for pre-computed FK data.
    
    The dataset loads FK samples from a .pt file containing:
    [joint_angles (n_joints), position (3), quaternion (4)]
    
    It handles normalization of joint angles and positions for training.
    """
    
    def __init__(self, dataset_path, metadata_path=None, pos_scale=1.0):
        """
        Initialize FK dataset.
        
        Args:
            dataset_path: Path to .pt file containing FK data
            metadata_path: Path to metadata JSON file (default: auto-detect from dataset_path)
            pos_scale: Scale factor for position normalization (default: 1.0)
        """
        self.dataset_path = Path(dataset_path)
        
        # Load dataset
        print(f"Loading dataset from: {self.dataset_path}")
        self.data = torch.load(self.dataset_path)
        print(f"Loaded {len(self.data)} samples")
        
        # Auto-detect metadata path
        if metadata_path is None:
            metadata_path = self.dataset_path.parent / f"{self.dataset_path.stem}_metadata.json"
        
        # Load metadata
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata from: {self.metadata_path}")
        else:
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # Extract metadata
        self.n_joints = self.metadata['n_joints']
        self.joint_names = self.metadata['joint_names']
        
        # Joint limits for normalization
        joint_limits = self.metadata['joint_limits']
        q_min = []
        q_max = []
        for joint_name in self.joint_names:
            lower, upper = joint_limits[joint_name]
            q_min.append(lower)
            q_max.append(upper)
        
        self.q_min = torch.tensor(q_min, dtype=torch.float32)
        self.q_max = torch.tensor(q_max, dtype=torch.float32)
        
        # Position scale
        self.pos_scale = pos_scale
        
        # Split data
        self.joint_angles = self.data[:, :self.n_joints]
        self.positions = self.data[:, self.n_joints:self.n_joints+3]
        self.quaternions = self.data[:, self.n_joints+3:self.n_joints+7]
        
        # Normalize joint angles to [-1, 1]
        self.joint_angles_norm = 2 * (self.joint_angles - self.q_min) / (self.q_max - self.q_min) - 1
        
        # Normalize positions
        self.positions_norm = self.positions / self.pos_scale
        
        # Normalize quaternions (should already be normalized, but ensure)
        self.quaternions_norm = self.quaternions / (torch.norm(self.quaternions, dim=-1, keepdim=True) + 1e-8)
        
        # Combine normalized pose: [pos_norm (3), quat_norm (4)]
        self.poses = torch.cat([self.positions_norm, self.quaternions_norm], dim=-1)
        
        print(f"Dataset info:")
        print(f"  Number of joints: {self.n_joints}")
        print(f"  Joint names: {self.joint_names}")
        print(f"  Number of samples: {len(self)}")
        print(f"  Joint angles range: [{self.joint_angles.min():.3f}, {self.joint_angles.max():.3f}]")
        print(f"  Position range: [{self.positions.min():.3f}, {self.positions.max():.3f}]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            joint_angles_norm: Normalized joint angles in [-1, 1]
            pose: [position_norm (3), quaternion (4)]
        """
        return self.joint_angles_norm[idx], self.poses[idx]
    
    def get_unnormalized(self, idx):
        """
        Get unnormalized data for a sample.
        
        Returns:
            joint_angles: Raw joint angles
            position: Raw position
            quaternion: Raw quaternion
        """
        return self.joint_angles[idx], self.positions[idx], self.quaternions[idx]
    
    def denormalize_joints(self, joint_angles_norm):
        """
        Convert normalized joint angles [-1, 1] back to raw joint angles.
        
        Args:
            joint_angles_norm: Tensor of shape (..., n_joints) in [-1, 1]
            
        Returns:
            joint_angles: Raw joint angles in radians
        """
        return (joint_angles_norm + 1) * (self.q_max - self.q_min) / 2 + self.q_min
    
    def denormalize_position(self, position_norm):
        """
        Convert normalized position back to raw position.
        
        Args:
            position_norm: Tensor of shape (..., 3)
            
        Returns:
            position: Raw position
        """
        return position_norm * self.pos_scale
    
    def get_dataloader(self, batch_size, shuffle=True, num_workers=0, pin_memory=False):
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )


def load_fk_dataset(dataset_path, metadata_path=None, pos_scale=1.0):
    """
    Convenience function to load an FK dataset.
    
    Args:
        dataset_path: Path to .pt file
        metadata_path: Path to metadata JSON (default: auto-detect)
        pos_scale: Position normalization scale
        
    Returns:
        FKDataset instance
    """
    return FKDataset(dataset_path, metadata_path, pos_scale)


def generate_dataset(
    urdf_path,
    output_path,
    num_samples=1000000,
    batch_size=10000,
    device=None,
    end_link=None,
    verbose=True
):
    """
    Generate FK dataset from URDF file.
    
    Args:
        urdf_path: Path to URDF file
        output_path: Path to save .pt dataset file
        num_samples: Number of FK samples to generate
        batch_size: Batch size for FK computation
        device: Device to use ('cuda' or 'cpu', None = auto)
        end_link: End effector link name (None = auto-detect)
        verbose: Whether to print progress
    
    Returns:
        Path to generated dataset
    """
    from mfik.robot_fk import RobotFK
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print("="*80)
        print("GENERATING FK DATASET")
        print("="*80)
        print(f"URDF: {urdf_path}")
        print(f"Output: {output_path}")
        print(f"Samples: {num_samples:,}")
        print(f"Device: {device}")
        print("="*80)
    
    # Load robot
    robot = RobotFK(urdf_path, device=device, end_link=end_link)
    if verbose:
        print(f"\nRobot loaded:")
        print(f"  Joints: {robot.n_joints}")
        print(f"  Joint names: {robot.actuated_joint_names}")
        print(f"  End link: {robot.end_link}")
        print(f"  Joint limits: [{robot.q_min.cpu().numpy()}, {robot.q_max.cpu().numpy()}]")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate samples in batches
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    if verbose:
        print(f"\nGenerating {num_samples:,} FK samples...")
    start_time = time.time()
    
    iterator = tqdm(range(num_batches), desc="Generating FK") if verbose else range(num_batches)
    for i in iterator:
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        # Sample random joint angles within limits
        q = torch.rand(current_batch_size, robot.n_joints, device=device)
        q = q * (robot.q_max - robot.q_min) + robot.q_min
        
        # Compute FK
        with torch.no_grad():
            pos, quat = robot.forward_kinematics(q)
        
        # Combine: [joint_angles (n), position (3), quaternion (4)]
        sample = torch.cat([q, pos, quat], dim=-1)
        all_samples.append(sample.cpu())
    
    # Concatenate all samples
    dataset = torch.cat(all_samples, dim=0)[:num_samples]
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nGeneration complete in {elapsed:.2f}s ({num_samples/elapsed:.0f} samples/sec)")
    
    # Save dataset
    torch.save(dataset, output_path)
    if verbose:
        print(f"Dataset saved to: {output_path}")
        print(f"Dataset shape: {dataset.shape}")
    
    # Save metadata
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    
    joint_limits = {}
    for i, name in enumerate(robot.actuated_joint_names):
        joint_limits[name] = [
            float(robot.q_min[i].cpu().item()),
            float(robot.q_max[i].cpu().item())
        ]
    
    metadata = {
        'urdf_path': str(urdf_path),
        'n_joints': robot.n_joints,
        'joint_names': robot.actuated_joint_names,
        'joint_limits': joint_limits,
        'end_link': robot.end_link.name,
        'num_samples': len(dataset),
        'data_format': 'joint_angles (n_joints), position (3), quaternion (4)'
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"Metadata saved to: {metadata_path}")
    
    # Print statistics
    if verbose:
        print("\nDataset statistics:")
        print(f"  Joint angles range: [{dataset[:, :robot.n_joints].min():.3f}, {dataset[:, :robot.n_joints].max():.3f}]")
        print(f"  Position range: [{dataset[:, robot.n_joints:robot.n_joints+3].min():.3f}, {dataset[:, robot.n_joints:robot.n_joints+3].max():.3f}]")
    
    return output_path


if __name__ == "__main__":
    # Test loading a dataset
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    print("="*80)
    print("Testing FKDataset")
    print("="*80)
    
    # Load dataset
    dataset = load_fk_dataset(dataset_path)
    
    # Test getting samples
    print("\n" + "="*80)
    print("Sample data:")
    print("="*80)
    
    for i in range(min(3, len(dataset))):
        joint_angles_norm, pose = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Normalized joint angles: {joint_angles_norm}")
        print(f"  Pose (pos_norm + quat): {pose}")
        
        # Test unnormalized
        joint_angles, position, quaternion = dataset.get_unnormalized(i)
        print(f"  Raw joint angles: {joint_angles}")
        print(f"  Raw position: {position}")
        print(f"  Raw quaternion: {quaternion}")
    
    # Test DataLoader
    print("\n" + "="*80)
    print("Testing DataLoader:")
    print("="*80)
    
    dataloader = dataset.get_dataloader(batch_size=64, shuffle=True)
    
    for batch_idx, (joint_angles_batch, poses_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}: joint_angles shape={joint_angles_batch.shape}, poses shape={poses_batch.shape}")
        if batch_idx >= 2:
            break
    
    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)
