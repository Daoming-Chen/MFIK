"""
Evaluation and visualization utilities for MeanFlow IK models.

Features:
- Model evaluation with comprehensive metrics
- Error distribution visualization
- Workspace coverage analysis
- Per-joint error analysis
- Interactive 3D visualization of predictions
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json
import time
from typing import Optional, Dict, Any, Tuple

from mfik.model import MeanFlowNetwork
from mfik.robot_fk import RobotFK


class Evaluator:
    """
    Comprehensive evaluator for MeanFlow IK models.
    
    Provides:
    - Accuracy metrics (position error, orientation error, success rates)
    - Error distribution analysis
    - Visualization of results
    - Workspace coverage analysis
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        urdf_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            urdf_path: Path to robot URDF file
            device: Device to use ('cuda' or 'cpu', None = auto)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load checkpoint
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = self.checkpoint['config']
        self.metadata = self.checkpoint['dataset_metadata']
        
        # Create and load model
        self.n_joints = self.metadata['n_joints']
        self.model = MeanFlowNetwork(
            state_dim=self.n_joints,
            condition_dim=7,
            hidden_dim=self.config.get('hidden_dim', 512),
            depth=self.config.get('depth', 6)
        ).to(device)
        
        # Load weights (prefer EMA model)
        if 'ema_model_state_dict' in self.checkpoint:
            self.model.load_state_dict(self.checkpoint['ema_model_state_dict'])
            self.using_ema = True
        else:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.using_ema = False
        
        self.model.eval()
        
        # Load robot FK
        self.urdf_path = urdf_path
        self.robot_fk = RobotFK(urdf_path, device=device)
        
        # Get joint limits from metadata
        q_min, q_max = [], []
        for joint_name in self.metadata['joint_names']:
            lower, upper = self.metadata['joint_limits'][joint_name]
            q_min.append(lower)
            q_max.append(upper)
        self.q_min = torch.tensor(q_min, device=device, dtype=torch.float32)
        self.q_max = torch.tensor(q_max, device=device, dtype=torch.float32)
        
        # Results storage
        self.results = None
        self.detailed_results = None
    
    def info(self):
        """Print model and robot information."""
        print("="*60)
        print("EVALUATOR INFO")
        print("="*60)
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Using EMA model: {self.using_ema}")
        print(f"Device: {self.device}")
        print()
        print("Model Config:")
        print(f"  Hidden dim: {self.config.get('hidden_dim')}")
        print(f"  Depth: {self.config.get('depth')}")
        print(f"  Mode: {self.config.get('mode')}")
        print()
        print("Robot Info:")
        print(f"  URDF: {self.urdf_path}")
        print(f"  Joints: {self.n_joints}")
        print(f"  Joint names: {self.metadata['joint_names']}")
        print("="*60)
    
    @torch.no_grad()
    def evaluate(
        self,
        num_samples: int = 1000,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Run evaluation on random samples.
        
        Args:
            num_samples: Number of test samples
            return_details: Whether to return detailed per-sample results
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Generate test samples
        q_gt = torch.rand(num_samples, self.n_joints, device=self.device)
        q_gt = q_gt * (self.q_max - self.q_min) + self.q_min
        
        # Compute ground truth FK
        target_pos, target_quat = self.robot_fk.forward_kinematics(q_gt)
        target_quat = target_quat / (torch.norm(target_quat, dim=-1, keepdim=True) + 1e-8)
        
        # Prepare target pose for model
        target_pose = torch.cat([target_pos, target_quat], dim=-1)
        
        # Inference
        start_time = time.time()
        
        # Sample from noise
        z1 = torch.randn(num_samples, self.n_joints, device=self.device)
        r = torch.zeros(num_samples, device=self.device)
        t = torch.ones(num_samples, device=self.device)
        
        u, _ = self.model(z1, r, t, target_pose)
        q_pred_norm = z1 - u
        
        # Denormalize
        q_pred = (q_pred_norm + 1) * (self.q_max - self.q_min) / 2 + self.q_min
        
        # Compute FK for predictions
        pred_pos, pred_quat = self.robot_fk.forward_kinematics(q_pred)
        pred_quat = pred_quat / (torch.norm(pred_quat, dim=-1, keepdim=True) + 1e-8)
        
        inference_time = time.time() - start_time
        
        # Compute errors
        pos_err = torch.norm(pred_pos - target_pos, dim=-1)
        
        # Orientation error (using quaternion distance)
        dot_prod = torch.abs(torch.sum(pred_quat * target_quat, dim=-1))
        dot_prod = torch.clamp(dot_prod, 0.0, 1.0)
        rot_err_rad = 2 * torch.acos(dot_prod)  # in radians
        rot_err_deg = torch.rad2deg(rot_err_rad)  # in degrees
        
        # Joint angle error (for diversity analysis)
        joint_err = q_pred - q_gt
        
        # Success metrics
        success_1cm = ((pos_err < 0.01) & (rot_err_rad < 0.1)).float().mean().item() * 100
        success_5mm = ((pos_err < 0.005) & (rot_err_rad < 0.05)).float().mean().item() * 100
        success_1mm = ((pos_err < 0.001) & (rot_err_rad < 0.01)).float().mean().item() * 100
        
        # Position-only success rates
        pos_success_1cm = (pos_err < 0.01).float().mean().item() * 100
        pos_success_5mm = (pos_err < 0.005).float().mean().item() * 100
        pos_success_1mm = (pos_err < 0.001).float().mean().item() * 100
        
        self.results = {
            'num_samples': num_samples,
            'inference_time_ms': inference_time * 1000,
            'per_sample_time_ms': inference_time / num_samples * 1000,
            
            # Position errors (in meters and cm)
            'pos_err_mean_m': pos_err.mean().item(),
            'pos_err_std_m': pos_err.std().item(),
            'pos_err_max_m': pos_err.max().item(),
            'pos_err_median_m': pos_err.median().item(),
            'pos_err_mean_cm': pos_err.mean().item() * 100,
            'pos_err_std_cm': pos_err.std().item() * 100,
            
            # Orientation errors (in degrees)
            'rot_err_mean_deg': rot_err_deg.mean().item(),
            'rot_err_std_deg': rot_err_deg.std().item(),
            'rot_err_max_deg': rot_err_deg.max().item(),
            'rot_err_median_deg': rot_err_deg.median().item(),
            
            # Success rates
            'success_1cm_0.1rad': success_1cm,
            'success_5mm_0.05rad': success_5mm,
            'success_1mm_0.01rad': success_1mm,
            'pos_success_1cm': pos_success_1cm,
            'pos_success_5mm': pos_success_5mm,
            'pos_success_1mm': pos_success_1mm,
        }
        
        if return_details:
            self.detailed_results = {
                'q_gt': q_gt.cpu(),
                'q_pred': q_pred.cpu(),
                'target_pos': target_pos.cpu(),
                'target_quat': target_quat.cpu(),
                'pred_pos': pred_pos.cpu(),
                'pred_quat': pred_quat.cpu(),
                'pos_err': pos_err.cpu(),
                'rot_err_deg': rot_err_deg.cpu(),
                'joint_err': joint_err.cpu(),
            }
        
        return self.results
    
    def print_results(self):
        """Print evaluation results in a formatted way."""
        if self.results is None:
            print("No evaluation results. Run evaluate() first.")
            return
        
        r = self.results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Samples: {r['num_samples']}")
        print(f"Total inference time: {r['inference_time_ms']:.2f} ms")
        print(f"Per-sample time: {r['per_sample_time_ms']:.4f} ms")
        print()
        print("Position Error:")
        print(f"  Mean:   {r['pos_err_mean_cm']:.4f} cm")
        print(f"  Std:    {r['pos_err_std_cm']:.4f} cm")
        print(f"  Median: {r['pos_err_median_m']*100:.4f} cm")
        print(f"  Max:    {r['pos_err_max_m']*100:.4f} cm")
        print()
        print("Orientation Error:")
        print(f"  Mean:   {r['rot_err_mean_deg']:.4f}°")
        print(f"  Std:    {r['rot_err_std_deg']:.4f}°")
        print(f"  Median: {r['rot_err_median_deg']:.4f}°")
        print(f"  Max:    {r['rot_err_max_deg']:.4f}°")
        print()
        print("Success Rates (Position + Orientation):")
        print(f"  <1cm, <5.7°:   {r['success_1cm_0.1rad']:.2f}%")
        print(f"  <5mm, <2.9°:   {r['success_5mm_0.05rad']:.2f}%")
        print(f"  <1mm, <0.6°:   {r['success_1mm_0.01rad']:.2f}%")
        print()
        print("Position-Only Success Rates:")
        print(f"  <1cm:  {r['pos_success_1cm']:.2f}%")
        print(f"  <5mm:  {r['pos_success_5mm']:.2f}%")
        print(f"  <1mm:  {r['pos_success_1mm']:.2f}%")
        print("="*60)
    
    def visualize_error_distribution(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Visualize error distributions.
        
        Args:
            save_path: Path to save figure (None = show)
            figsize: Figure size
        """
        if self.detailed_results is None:
            print("No detailed results. Run evaluate(return_details=True) first.")
            return
        
        pos_err = self.detailed_results['pos_err'].numpy() * 100  # Convert to cm
        rot_err = self.detailed_results['rot_err_deg'].numpy()
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig)
        
        # Position error histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(pos_err, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax1.axvline(np.mean(pos_err), color='red', linestyle='--', label=f'Mean: {np.mean(pos_err):.3f} cm')
        ax1.axvline(np.median(pos_err), color='orange', linestyle='--', label=f'Median: {np.median(pos_err):.3f} cm')
        ax1.set_xlabel('Position Error (cm)')
        ax1.set_ylabel('Count')
        ax1.set_title('Position Error Distribution')
        ax1.legend()
        
        # Orientation error histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(rot_err, bins=50, color='coral', edgecolor='white', alpha=0.8)
        ax2.axvline(np.mean(rot_err), color='red', linestyle='--', label=f'Mean: {np.mean(rot_err):.3f}°')
        ax2.axvline(np.median(rot_err), color='orange', linestyle='--', label=f'Median: {np.median(rot_err):.3f}°')
        ax2.set_xlabel('Orientation Error (degrees)')
        ax2.set_ylabel('Count')
        ax2.set_title('Orientation Error Distribution')
        ax2.legend()
        
        # Position vs Orientation error scatter
        ax3 = fig.add_subplot(gs[0, 2])
        scatter = ax3.scatter(pos_err, rot_err, c=pos_err + rot_err/10, cmap='viridis', 
                             alpha=0.5, s=10)
        ax3.axvline(1.0, color='green', linestyle='--', alpha=0.5, label='1 cm threshold')
        ax3.axhline(5.7, color='green', linestyle='--', alpha=0.5, label='5.7° threshold')
        ax3.set_xlabel('Position Error (cm)')
        ax3.set_ylabel('Orientation Error (degrees)')
        ax3.set_title('Position vs Orientation Error')
        ax3.legend()
        plt.colorbar(scatter, ax=ax3, label='Combined Error')
        
        # CDF plots
        ax4 = fig.add_subplot(gs[1, 0])
        sorted_pos = np.sort(pos_err)
        cdf_pos = np.arange(1, len(sorted_pos) + 1) / len(sorted_pos) * 100
        ax4.plot(sorted_pos, cdf_pos, color='steelblue', linewidth=2)
        ax4.axvline(1.0, color='green', linestyle='--', alpha=0.7)
        ax4.axvline(0.5, color='orange', linestyle='--', alpha=0.7)
        ax4.axvline(0.1, color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Position Error (cm)')
        ax4.set_ylabel('Cumulative Percentage (%)')
        ax4.set_title('Position Error CDF')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, np.percentile(pos_err, 99))
        
        ax5 = fig.add_subplot(gs[1, 1])
        sorted_rot = np.sort(rot_err)
        cdf_rot = np.arange(1, len(sorted_rot) + 1) / len(sorted_rot) * 100
        ax5.plot(sorted_rot, cdf_rot, color='coral', linewidth=2)
        ax5.axvline(5.7, color='green', linestyle='--', alpha=0.7, label='5.7° (0.1 rad)')
        ax5.axvline(2.9, color='orange', linestyle='--', alpha=0.7, label='2.9° (0.05 rad)')
        ax5.set_xlabel('Orientation Error (degrees)')
        ax5.set_ylabel('Cumulative Percentage (%)')
        ax5.set_title('Orientation Error CDF')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.set_xlim(0, np.percentile(rot_err, 99))
        
        # Per-joint error boxplot
        ax6 = fig.add_subplot(gs[1, 2])
        joint_err = self.detailed_results['joint_err'].numpy()
        joint_names = [f'J{i+1}' for i in range(self.n_joints)]
        bp = ax6.boxplot(joint_err, labels=joint_names, patch_artist=True)
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_joints))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax6.set_xlabel('Joint')
        ax6.set_ylabel('Joint Angle Error (rad)')
        ax6.set_title('Per-Joint Angle Error')
        ax6.axhline(0, color='gray', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_workspace(
        self,
        num_samples: int = 500,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Visualize workspace coverage and prediction accuracy in 3D.
        
        Args:
            num_samples: Number of samples to visualize
            save_path: Path to save figure (None = show)
            figsize: Figure size
        """
        if self.detailed_results is None:
            print("No detailed results. Run evaluate(return_details=True) first.")
            return
        
        # Limit samples for visualization
        n = min(num_samples, len(self.detailed_results['target_pos']))
        
        target_pos = self.detailed_results['target_pos'][:n].numpy()
        pred_pos = self.detailed_results['pred_pos'][:n].numpy()
        pos_err = self.detailed_results['pos_err'][:n].numpy() * 100  # cm
        
        fig = plt.figure(figsize=figsize)
        
        # 3D scatter - target positions colored by error
        ax1 = fig.add_subplot(121, projection='3d')
        scatter = ax1.scatter(
            target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
            c=pos_err, cmap='RdYlGn_r', s=20, alpha=0.7,
            vmin=0, vmax=np.percentile(pos_err, 95)
        )
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Workspace Coverage\n(colored by position error)')
        plt.colorbar(scatter, ax=ax1, label='Error (cm)', shrink=0.6)
        
        # 3D scatter - prediction vs target with error vectors
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Sample fewer points for error vectors to avoid clutter
        step = max(1, n // 100)
        for i in range(0, n, step):
            ax2.plot(
                [target_pos[i, 0], pred_pos[i, 0]],
                [target_pos[i, 1], pred_pos[i, 1]],
                [target_pos[i, 2], pred_pos[i, 2]],
                color='red', alpha=0.3, linewidth=0.5
            )
        
        ax2.scatter(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
                   c='blue', s=10, alpha=0.5, label='Target')
        ax2.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2],
                   c='green', s=10, alpha=0.5, label='Predicted')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('Target vs Predicted Positions\n(red lines = errors)')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_all(
        self,
        output_dir: Optional[str] = None,
        prefix: str = "eval"
    ):
        """
        Generate all visualizations.
        
        Args:
            output_dir: Directory to save figures (None = show interactively)
            prefix: Filename prefix for saved figures
        """
        if self.detailed_results is None:
            print("Running evaluation with detailed results...")
            self.evaluate(num_samples=1000, return_details=True)
        
        self.print_results()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.visualize_error_distribution(
                save_path=str(output_dir / f"{prefix}_error_distribution.png")
            )
            self.visualize_workspace(
                save_path=str(output_dir / f"{prefix}_workspace.png")
            )
            
            # Save results as JSON
            results_path = output_dir / f"{prefix}_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to: {results_path}")
        else:
            self.visualize_error_distribution()
            self.visualize_workspace()
    
    def compare_refinement(
        self,
        num_samples: int = 100,
        refinement_steps: int = 10,
        refinement_lr: float = 0.05,
        save_path: Optional[str] = None
    ):
        """
        Compare model predictions with and without refinement.
        
        Args:
            num_samples: Number of test samples
            refinement_steps: Number of refinement iterations
            refinement_lr: Learning rate for refinement
            save_path: Path to save figure
        """
        # Generate test samples
        q_gt = torch.rand(num_samples, self.n_joints, device=self.device)
        q_gt = q_gt * (self.q_max - self.q_min) + self.q_min
        
        with torch.no_grad():
            target_pos, target_quat = self.robot_fk.forward_kinematics(q_gt)
            target_quat = target_quat / (torch.norm(target_quat, dim=-1, keepdim=True) + 1e-8)
        
        target_pose = torch.cat([target_pos, target_quat], dim=-1)
        
        # Initial prediction
        with torch.no_grad():
            z1 = torch.randn(num_samples, self.n_joints, device=self.device)
            r = torch.zeros(num_samples, device=self.device)
            t = torch.ones(num_samples, device=self.device)
            
            u, _ = self.model(z1, r, t, target_pose)
            q_init_norm = z1 - u
            q_init = (q_init_norm + 1) * (self.q_max - self.q_min) / 2 + self.q_min
        
        # Refinement
        q_refined = q_init.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([q_refined], lr=refinement_lr)
        
        errors_over_steps = []
        
        for step in range(refinement_steps + 1):
            with torch.no_grad():
                pred_pos, pred_quat = self.robot_fk.forward_kinematics(q_refined)
                pred_quat = pred_quat / (torch.norm(pred_quat, dim=-1, keepdim=True) + 1e-8)
                pos_err = torch.norm(pred_pos - target_pos, dim=-1)
                errors_over_steps.append(pos_err.mean().item() * 100)  # cm
            
            if step < refinement_steps:
                optimizer.zero_grad()
                pred_pos, pred_quat = self.robot_fk.forward_kinematics(q_refined)
                pred_quat = pred_quat / (torch.norm(pred_quat, dim=-1, keepdim=True) + 1e-8)
                
                loss = torch.norm(pred_pos - target_pos, dim=-1).mean()
                loss += torch.norm(pred_quat - target_quat, dim=-1).mean()
                
                loss.backward()
                optimizer.step()
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(refinement_steps + 1), errors_over_steps, 'b-o', linewidth=2, markersize=8)
        ax.axhline(errors_over_steps[0], color='red', linestyle='--', 
                   label=f'Initial: {errors_over_steps[0]:.4f} cm')
        ax.axhline(errors_over_steps[-1], color='green', linestyle='--',
                   label=f'Final: {errors_over_steps[-1]:.4f} cm')
        
        ax.set_xlabel('Refinement Step')
        ax.set_ylabel('Mean Position Error (cm)')
        ax.set_title('Error Reduction with Refinement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        improvement = (1 - errors_over_steps[-1] / errors_over_steps[0]) * 100
        ax.text(0.5, 0.95, f'Improvement: {improvement:.1f}%', 
                transform=ax.transAxes, ha='center', va='top',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return {
            'initial_error_cm': errors_over_steps[0],
            'final_error_cm': errors_over_steps[-1],
            'improvement_percent': improvement,
            'errors_over_steps': errors_over_steps
        }


def evaluate_model(
    checkpoint_path: str,
    urdf_path: str,
    num_samples: int = 1000,
    device: Optional[str] = None,
    visualize: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        urdf_path: Path to URDF file
        num_samples: Number of test samples
        device: Device to use
        visualize: Whether to generate visualizations
        output_dir: Directory to save outputs (None = show interactively)
    
    Returns:
        Evaluation results dictionary
    """
    evaluator = Evaluator(checkpoint_path, urdf_path, device)
    evaluator.info()
    
    results = evaluator.evaluate(num_samples, return_details=True)
    evaluator.print_results()
    
    if visualize:
        evaluator.visualize_all(output_dir)
    
    return results


def main():
    """Command-line interface for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MeanFlow IK Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--urdf', type=str, required=True, help='Path to URDF file')
    parser.add_argument('--samples', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save outputs')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization')
    parser.add_argument('--refinement-test', action='store_true', help='Test refinement')
    
    args = parser.parse_args()
    
    evaluator = Evaluator(args.checkpoint, args.urdf, args.device)
    evaluator.info()
    
    # Main evaluation
    results = evaluator.evaluate(args.samples, return_details=True)
    evaluator.print_results()
    
    # Visualizations
    if not args.no_visualize:
        evaluator.visualize_all(args.output_dir)
    
    # Refinement test
    if args.refinement_test:
        print("\n" + "="*60)
        print("REFINEMENT TEST")
        print("="*60)
        save_path = None
        if args.output_dir:
            save_path = str(Path(args.output_dir) / "refinement_test.png")
        refinement_results = evaluator.compare_refinement(save_path=save_path)
        print(f"Improvement with refinement: {refinement_results['improvement_percent']:.1f}%")


if __name__ == "__main__":
    main()
