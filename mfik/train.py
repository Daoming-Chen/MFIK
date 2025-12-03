import torch
import torch.optim as optim
from torch.func import jvp, functional_call
from copy import deepcopy
from pathlib import Path
from mfik.model import MeanFlowNetwork
from mfik.dataset import FKDataset
import time

class Trainer:
    def __init__(self, config, dataset=None):
        """
        Initialize Trainer.
        
        Args:
            config: Configuration dictionary
            dataset: FKDataset instance (if None, will create from config['dataset_path'])
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        if dataset is None:
            dataset_path = config.get('dataset_path')
            if dataset_path is None:
                raise ValueError("Either 'dataset' or 'dataset_path' must be provided in config")
            pos_scale = config.get('pos_scale', 1.0)
            self.dataset = FKDataset(dataset_path, pos_scale=pos_scale)
        else:
            self.dataset = dataset
        
        # Get robot info from dataset
        self.n_joints = self.dataset.n_joints
        self.q_min = self.dataset.q_min.to(self.device)
        self.q_max = self.dataset.q_max.to(self.device)
        self.pos_scale = self.dataset.pos_scale
        
        # Model Setup
        # State dim = n_joints
        # Condition dim = 7 (3 pos + 4 quat)
        self.model = MeanFlowNetwork(
            state_dim=self.n_joints,
            condition_dim=7,
            hidden_dim=config.get('hidden_dim', 512),
            depth=config.get('depth', 6)
        ).to(self.device)
        
        # Initialize EMA model
        self.ema_model = deepcopy(self.model).eval().requires_grad_(False)
        self.ema_decay = config.get('ema_decay', 0.999)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('lr', 1e-4))
        
        self.batch_size = config.get('batch_size', 256)
        self.iterations = config.get('iterations', 10000)
        self.mode = config.get('mode', 'meanflow') # 'baseline' or 'meanflow'
        self.lambda_disp = config.get('lambda_disp', 0.01)  # Greatly reduced dispersive weight
        self.alpha_meanflow = config.get('alpha_meanflow', 0.1)  # Weight for meanflow term
        
        # Create DataLoader
        self.dataloader = self.dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            pin_memory=(self.device == 'cuda')
        )
        
        # Initialize data iterator
        self.data_iter = iter(self.dataloader)
    
    def update_ema(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def generate_batch(self):
        """
        Get a batch of data from the dataset.
        Uses pre-computed FK data instead of online computation.
        
        Returns:
            q_norm: Normalized joint angles [-1, 1]
            target_pose: [position_norm (3), quaternion (4)]
        """
        try:
            q_norm, target_pose = next(self.data_iter)
        except StopIteration:
            # Restart iterator when epoch ends
            self.data_iter = iter(self.dataloader)
            q_norm, target_pose = next(self.data_iter)
        
        # Move to device
        q_norm = q_norm.to(self.device)
        target_pose = target_pose.to(self.device)
        
        return q_norm, target_pose

    def dispersive_loss(self, features, tau=0.5):
        # features: [B, D]
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device)

        # Compute pairwise distance squared
        dist_sq = torch.cdist(features, features, p=2) ** 2
        
        # exp(-dist/tau)
        kernel = torch.exp(-dist_sq / tau)
        
        # Exclude diagonal
        mask = torch.eye(batch_size, device=features.device)
        sum_kernel = torch.sum(kernel * (1 - mask), dim=1)
        
        # log(mean over i of sum over j (j!=i))
        # Normalize by B-1
        loss = torch.log(torch.mean(sum_kernel / (batch_size - 1)) + 1e-8)
        return loss

    def train_step_fm_baseline(self):
        self.optimizer.zero_grad()
        x0, condition = self.generate_batch()
        x1 = torch.randn_like(x0)
        
        # Logit-Normal Sampling
        mu, sigma = 0.0, 1.0
        batch_size = x0.shape[0]
        t_normal = torch.randn(batch_size, device=self.device) * sigma + mu
        t = torch.sigmoid(t_normal)
        
        t_reshaped = t.view(-1, 1)
        
        z_t = (1 - t_reshaped) * x0 + t_reshaped * x1
        v_target = x1 - x0
        r = torch.zeros_like(t)
        
        v_pred, _ = self.model(z_t, r, t, condition)
        loss = torch.mean((v_pred - v_target) ** 2)
        
        loss.backward()
        self.optimizer.step()
        self.update_ema()
        return loss.item()

    def train_step_meanflow(self):
        self.optimizer.zero_grad()
        
        # Data (x0) and Noise (x1)
        x0, condition = self.generate_batch()
        x1 = torch.randn_like(x0)
        
        # Logit-Normal Sampling
        mu, sigma = 0.0, 1.0
        batch_size = x0.shape[0]
        t_normal = torch.randn(batch_size, device=self.device) * sigma + mu
        t = torch.sigmoid(t_normal)
        
        r = torch.zeros_like(t)  # Reference point at t=0
        
        t_reshaped = t.view(-1, 1)
        
        # Interpolate z_t = (1-t)x0 + t*x1
        z_t = (1 - t_reshaped) * x0 + t_reshaped * x1
        v_t = x1 - x0 # dx/dt
        
        # Standard flow matching loss (main objective)
        v_pred, features = self.model(z_t, r, t, condition)
        flow_loss = torch.mean((v_pred - v_t) ** 2)
        
        # MeanFlow consistency term (only if alpha > 0)
        meanflow_loss = torch.tensor(0.0, device=self.device)
        if self.alpha_meanflow > 0:
            # JVP Calculation - compute du/dt
            params = dict(self.model.named_parameters())
            buffers = dict(self.model.named_buffers())
            
            def forward_closure(z, ref_t, curr_t, c):
                 return functional_call(self.model, (params, buffers), (z, ref_t, curr_t, c))
                 
            # Only differentiate w.r.t time t
            tangents = (torch.zeros_like(z_t), torch.zeros_like(r), torch.ones_like(t), torch.zeros_like(condition))
            
            # jvp returns ((out, features), (out_dot, features_dot))
            (u_pred_no_grad, _), (du_dt, _) = jvp(forward_closure, (z_t, r, t, condition), tangents)
            
            # MeanFlow target: u should satisfy consistency
            # For flow from x0 to x1: u(t) ≈ v - t * du/dt (simplified)
            u_target = v_t - t_reshaped * du_dt
            u_target = u_target.detach()
            
            meanflow_loss = torch.mean((v_pred - u_target) ** 2)
        
        # Dispersive Loss (encourage diversity)
        disp_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_disp > 0 and self.batch_size > 1:
            disp_loss = self.dispersive_loss(features, tau=0.5)
        
        # Combined loss
        loss = flow_loss + self.alpha_meanflow * meanflow_loss + self.lambda_disp * disp_loss
        
        loss.backward()
        self.optimizer.step()
        self.update_ema()
        
        # Convert to Python scalars for logging
        return loss.item(), flow_loss.item(), meanflow_loss.item() if isinstance(meanflow_loss, torch.Tensor) else meanflow_loss, disp_loss.item() if isinstance(disp_loss, torch.Tensor) else disp_loss

    def refine_solution(self, q_init, target_pose, steps=10, lr=0.05):
        """
        Refine solution using gradient descent.
        Note: This requires a robot FK model for validation.
        Currently disabled - use with RobotFK when needed.
        """
        # For now, return as-is since we don't have robot FK loaded
        # In production, you'd load RobotFK with the URDF and compute FK
        return q_init

    def export_onnx(self, path="meanflow_ik.onnx"):
        self.ema_model.eval()
        # Dummy inputs
        z = torch.randn(1, 7, device=self.device)
        r = torch.zeros(1, device=self.device)
        t = torch.ones(1, device=self.device)
        cond = torch.randn(1, 7, device=self.device)
        
        torch.onnx.export(
            self.ema_model,
            (z, r, t, cond),
            path,
            input_names=["z", "r", "t", "condition"],
            output_names=["u", "features"],
            dynamic_axes={"z": {0: "batch"}, "condition": {0: "batch"}, "u": {0: "batch"}}
        )
        print(f"Model (EMA) exported to {path}")

    def validate(self, robot_fk=None):
        """
        Validate the model.
        
        Args:
            robot_fk: Optional RobotFK instance for computing actual FK errors.
                     If None, validation is skipped.
        """
        if robot_fk is None:
            print("\n" + "="*60)
            print("Validation skipped: No RobotFK provided")
            print("To enable validation, pass a RobotFK instance")
            print("="*60)
            return
        
        print("\n" + "="*60)
        print("Validating 1-NFE Inference (using EMA)...")
        self.ema_model.eval()
        
        # Sample validation data from dataset
        val_batch_size = min(1000, len(self.dataset))
        val_indices = torch.randperm(len(self.dataset))[:val_batch_size]
        
        q_gt_norm = torch.stack([self.dataset.joint_angles_norm[i] for i in val_indices])
        target_pose = torch.stack([self.dataset.poses[i] for i in val_indices])
        
        q_gt_norm = q_gt_norm.to(self.device)
        target_pose = target_pose.to(self.device)
        
        # Denormalize for FK computation
        q_gt = self.dataset.denormalize_joints(q_gt_norm)
        q_gt = q_gt.to(self.device)
        
        with torch.no_grad():
            # Compute ground truth FK
            target_pos, target_quat = robot_fk.forward_kinematics(q_gt)
            target_quat = target_quat / (torch.norm(target_quat, dim=-1, keepdim=True) + 1e-8)
            
            # Predict
            start_time = time.time()
            q_pred_norm = self.solve(target_pose, model=self.ema_model)
            end_time = time.time()
            
            # Denormalize predicted joints
            q_pred = self.dataset.denormalize_joints(q_pred_norm)
            
            # Compute FK for predictions
            pred_pos, pred_quat = robot_fk.forward_kinematics(q_pred)
            pred_quat = pred_quat / (torch.norm(pred_quat, dim=-1, keepdim=True) + 1e-8)
            
            pos_err = torch.norm(pred_pos - target_pos, dim=-1)
            
            # Orientation error
            dot_prod = torch.sum(pred_quat * target_quat, dim=-1)
            rot_err = 1.0 - dot_prod**2
            
            success_rate = ((pos_err < 0.01) & (rot_err < 0.01)).float().mean().item() * 100
            
            print(f"1-NFE Inference Speed: {(end_time - start_time)*1000:.4f} ms/batch")
            print(f"1-NFE Mean Pos Error: {pos_err.mean().item()*100:.2f} cm")
            print(f"1-NFE Mean Rot Error: {rot_err.mean().item():.4f}")
            print(f"1-NFE Success Rate (<1cm, <0.01 rot): {success_rate:.2f}%")
            
            # Diversity Metric
            if val_batch_size > 1:
                dists = torch.cdist(q_pred_norm, q_pred_norm, p=2)
                mask = ~torch.eye(val_batch_size, dtype=torch.bool, device=self.device)
                mean_dist = dists[mask].mean().item()
                print(f"1-NFE Diversity (Mean Joint Dist): {mean_dist:.4f}")

    def train(self, robot_fk=None):
        """
        Train the model.
        
        Args:
            robot_fk: Optional RobotFK instance for validation
        """
        print(f"\n{'='*60}")
        print(f"Starting Training ({self.mode})...")
        print(f"Config: hidden_dim={self.config.get('hidden_dim')}, depth={self.config.get('depth')}")
        print(f"        batch_size={self.batch_size}, lr={self.config.get('lr', 1e-4)}")
        print(f"        n_joints={self.n_joints}, dataset_size={len(self.dataset)}")
        if self.mode == 'meanflow':
            print(f"        alpha_meanflow={self.alpha_meanflow}, lambda_disp={self.lambda_disp}")
        print(f"{'='*60}\n")
        
        for i in range(self.iterations):
            if self.mode == 'meanflow':
                loss, flow_loss, mf_loss, disp_loss = self.train_step_meanflow()
                if i % 100 == 0:
                    print(f"Iter {i:5d}: Loss={loss:7.4f} [Flow={flow_loss:6.4f}, MF={mf_loss:6.4f}, Disp={disp_loss:7.3f}]")
            else:
                loss = self.train_step_fm_baseline()
                if i % 100 == 0:
                    print(f"Iter {i:5d}: Loss={loss:7.4f}")
        
        self.validate(robot_fk)
        self.export_onnx()
    
    @torch.no_grad()
    def solve(self, target_pose, model=None):
        """
        Solve IK for given target pose.
        Returns normalized joint angles in [-1, 1].
        """
        if model is None:
            model = self.model
        batch_size = target_pose.shape[0]
        z1 = torch.randn(batch_size, self.n_joints, device=self.device)
        
        r = torch.zeros(batch_size, device=self.device)
        t = torch.ones(batch_size, device=self.device)
        
        u, _ = model(z1, r, t, target_pose)
        
        z0_pred = z1 - u
        # Return normalized joint angles
        return z0_pred
        
if __name__ == "__main__":
    # Example usage - need to provide dataset path
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    config = {
        'dataset_path': dataset_path,
        'hidden_dim': 512, 
        'depth': 8,
        'batch_size': 1024,
        'iterations': 50000,
        'mode': 'meanflow'
    }
    trainer = Trainer(config)
    trainer.train()


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(trainer, scheduler, iteration, config, metadata, path):
    """Save training checkpoint."""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': trainer.model.state_dict(),
        'ema_model_state_dict': trainer.ema_model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config,
        'dataset_metadata': metadata
    }
    torch.save(checkpoint, path)


def load_checkpoint(checkpoint_path, trainer, scheduler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['iteration']


# ============================================================================
# High-level Training Function
# ============================================================================

def train_from_dataset(
    dataset_path,
    urdf_path=None,
    hidden_dim=1024,
    depth=12,
    batch_size=2048,
    iterations=100000,
    lr=2e-3,
    mode='meanflow',
    alpha_meanflow=0.1,
    lambda_disp=0.02,
    checkpoint_dir='checkpoints',
    device=None,
    validate=True,
    checkpoint_interval=10000,
    validation_interval=5000,
    verbose=True
):
    """
    Train MeanFlow IK model from pre-generated dataset.
    
    Args:
        dataset_path: Path to .pt dataset file
        urdf_path: Path to URDF file (for validation FK, optional)
        hidden_dim: Model hidden dimension
        depth: Model depth
        batch_size: Training batch size
        iterations: Number of training iterations
        lr: Learning rate
        mode: 'meanflow' or 'baseline'
        alpha_meanflow: MeanFlow consistency weight
        lambda_disp: Diversity loss weight
        checkpoint_dir: Directory to save checkpoints
        device: Device to use ('cuda' or 'cpu', None = auto)
        validate: Whether to run validation (requires urdf_path)
        checkpoint_interval: Save checkpoint every N iterations
        validation_interval: Run validation every N iterations
        verbose: Whether to print progress
    
    Returns:
        Trained Trainer instance
    """
    from mfik.robot_fk import RobotFK
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"Using device: {device}")
    
    # Create checkpoints directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    
    # Load dataset
    if verbose:
        print("\n" + "="*80)
        print("Loading Dataset")
        print("="*80)
    dataset = FKDataset(dataset_path)
    
    # Configuration
    config = {
        'hidden_dim': hidden_dim,
        'depth': depth,
        'batch_size': batch_size,
        'iterations': iterations,
        'lr': lr,
        'mode': mode,
        'alpha_meanflow': alpha_meanflow,
        'lambda_disp': lambda_disp,
        'device': device
    }
    
    if verbose:
        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        print(f"Dataset: {dataset_path}")
        print(f"  - Samples: {len(dataset):,}")
        print(f"  - Joints: {dataset.n_joints}")
        print(f"Model: hidden_dim={hidden_dim}, depth={depth}")
        print(f"Training: batch_size={batch_size}, iterations={iterations}")
        print(f"Learning rate: {lr}")
        if mode == 'meanflow':
            print(f"MeanFlow: alpha={alpha_meanflow}, lambda_disp={lambda_disp}")
        print(f"Device: {device}")
        print("="*80)
    
    # Create trainer
    trainer = Trainer(config, dataset=dataset)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, 
        T_max=iterations,
        eta_min=1e-5
    )
    
    # Load robot FK for validation if URDF provided
    robot_fk = None
    if validate and urdf_path is not None:
        if verbose:
            print(f"\nLoading robot FK from URDF: {urdf_path}")
        robot_fk = RobotFK(urdf_path, device=device)
        if verbose:
            print(f"Robot FK loaded: {robot_fk.n_joints} joints")
    elif validate and urdf_path is None and verbose:
        print("\nWARNING: Validation requested but no URDF path provided")
        print("Validation will be skipped")
    
    # Train
    if verbose:
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
    
    start_time = time.time()
    
    for i in range(iterations):
        # Training step
        if mode == 'meanflow':
            loss, flow_loss, mf_loss, disp_loss = trainer.train_step_meanflow()
        else:
            loss = trainer.train_step_fm_baseline()
        
        # Update learning rate
        scheduler.step()
        
        # Logging
        if verbose and i % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            iter_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            eta_seconds = (iterations - i - 1) / iter_per_sec if iter_per_sec > 0 else 0
            eta_mins = eta_seconds / 60
            
            if mode == 'meanflow':
                print(f"Iter {i:6d}/{iterations}: Loss={loss:7.4f} "
                      f"[Flow={flow_loss:6.4f}, MF={mf_loss:6.4f}, Disp={disp_loss:7.3f}] "
                      f"LR={current_lr:.2e} | {iter_per_sec:.1f} it/s | ETA: {eta_mins:.1f}m")
            else:
                print(f"Iter {i:6d}/{iterations}: Loss={loss:7.4f} "
                      f"LR={current_lr:.2e} | {iter_per_sec:.1f} it/s | ETA: {eta_mins:.1f}m")
        
        # Periodic validation
        if robot_fk is not None and i > 0 and i % validation_interval == 0:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Validation at iteration {i}")
                print(f"{'='*60}")
            trainer.validate(robot_fk)
            if verbose:
                print()
        
        # Save checkpoint
        if i > 0 and i % checkpoint_interval == 0:
            ckpt_file = checkpoint_path / f"checkpoint_iter_{i}.pt"
            save_checkpoint(trainer, scheduler, i, config, dataset.metadata, ckpt_file)
            if verbose:
                print(f"Checkpoint saved: {ckpt_file}\n")
    
    end_time = time.time()
    
    if verbose:
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
    
    # Final validation
    if robot_fk is not None:
        if verbose:
            print("\nFinal Validation:")
        trainer.validate(robot_fk)
    
    # Save final checkpoint
    checkpoint_file = checkpoint_path / f"meanflow_ik_{mode}_final.pt"
    save_checkpoint(trainer, scheduler, iterations, config, dataset.metadata, checkpoint_file)
    if verbose:
        print(f"\nFinal model saved to: {checkpoint_file}")
    
    # Export ONNX
    onnx_path = checkpoint_path / f"meanflow_ik_{mode}.onnx"
    trainer.export_onnx(str(onnx_path))
    
    return trainer