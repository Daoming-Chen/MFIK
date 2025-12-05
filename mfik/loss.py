import torch
import torch.nn as nn
from torch.func import jvp
import numpy as np


class MeanFlowLoss:
    """
    MeanFlow Loss with Neighborhood Projection for IK.
    
    Key insight from method.md:
    - Instead of Flow from random noise to data, we Flow from q_ref (perturbed q_gt) to q_gt
    - This converts the one-to-many IK problem into a one-to-one projection problem
    - At t=0: q_ref (noisy reference joint config)
    - At t=1: q_gt (ground truth solution)
    - The network learns to project q_ref to the nearest valid IK solution
    """
    
    def __init__(
        self,
        time_sampler="uniform",
        time_mu=-0.4,
        time_sigma=1.0,
        ratio_r_not_equal_t=0.0,  # Pure Flow Matching when 0
        adaptive_p=1.0,
        weighting="uniform",
        # Neighborhood projection noise parameters
        noise_std_min=0.1,   # Minimum noise std for q_ref generation
        noise_std_max=1.0,   # Maximum noise std for q_ref generation  
        curriculum_enabled=True,  # Whether to use curriculum learning for noise
    ):
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.adaptive_p = adaptive_p
        self.weighting = weighting
        # Neighborhood projection params
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.curriculum_enabled = curriculum_enabled
        self.current_epoch = 0
        self.total_epochs = 100

    def set_epoch(self, epoch, total_epochs):
        """Update current epoch for curriculum learning"""
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def get_noise_std(self, batch_size, device):
        """
        Get noise standard deviation for q_ref generation.
        Uses curriculum learning: start with large noise, decrease over training.
        """
        if self.curriculum_enabled and self.total_epochs > 0:
            # Progress from 0 to 1
            progress = self.current_epoch / self.total_epochs
            # Noise decreases from max to min over training
            # Use cosine schedule for smooth decay
            noise_std = self.noise_std_min + 0.5 * (self.noise_std_max - self.noise_std_min) * (1 + np.cos(np.pi * progress))
        else:
            # Random noise in range
            noise_std = torch.rand(batch_size, device=device) * (self.noise_std_max - self.noise_std_min) + self.noise_std_min
            return noise_std
        
        return torch.full((batch_size,), noise_std, device=device)

    def sample_time_steps(self, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step 1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Step 2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Step 3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        r = torch.where(equal_mask, t, r)
        
        return r, t

    def __call__(self, model, joints, pose):
        """
        Compute MeanFlow loss with Neighborhood Projection.
        
        Flow direction: q_ref (t=0) -> q_gt (t=1)
        - q_ref = q_gt + noise (reference joint from neighborhood)
        - q_gt = ground truth solution
        
        Args:
            model: The conditional MLP model
            joints: (B, n_joints) Ground truth joint configurations (q_gt)
            pose: (B, 7) Condition (end-effector pose)
        """
        batch_size = joints.shape[0]
        device = joints.device
        
        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device)  # (B,), (B,)
        
        # Generate q_ref by adding noise to q_gt (Neighborhood Projection)
        # q_ref = q_gt + noise, where noise ~ N(0, sigma^2)
        noise_std = self.get_noise_std(batch_size, device)  # (B,)
        noise = torch.randn_like(joints) * noise_std.view(-1, 1)
        q_ref = joints + noise  # This is our starting point at t=0
        
        # q_gt is our target at t=1 (the ground truth joints)
        q_gt = joints
        
        # Velocity field: v = q_gt - q_ref (direction from q_ref to q_gt)
        v_t = q_gt - q_ref
        
        # Linear interpolation: z_t = q_ref + t * v_t = q_ref + t * (q_gt - q_ref)
        # At t=0: z_t = q_ref
        # At t=1: z_t = q_gt
        t_b = t.view(-1, 1)
        z_t = q_ref + t_b * v_t
        
        # Time difference for update
        time_diff = (t - r).view(-1, 1)
        
        # Forward pass for u (prediction at z_t, r, t, noise_std)
        # noise_std is passed as the "condition" c to help model understand projection radius
        u = model(z_t, pose, r, t, noise_std)
        
        # Calculate JVP for target velocity u_target
        # u_target = v_t - (t-r) * du/dt
        
        def model_func(z, curr_r, curr_t):
            return model(z, pose, curr_r, curr_t, noise_std)
            
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
        
        # Compute JVP
        _, dudt = jvp(model_func, primals, tangents)
        
        # Target velocity
        u_target = v_t - time_diff * dudt
        
        # Compute Loss
        # Detach target to stop gradients flowing through target calculation
        diff = u - u_target.detach()
        loss_per_sample = torch.sum(diff**2, dim=-1)  # Sum over joints
        
        if self.weighting == "adaptive":
            weights = 1.0 / (loss_per_sample.detach() + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_per_sample
        else:
            loss = loss_per_sample
            
        return loss.mean(), loss_per_sample.mean()


if __name__ == "__main__":
    # Test
    from mfik.model import ConditionalMLP
    
    batch_size = 4
    n_joints = 7
    model = ConditionalMLP(n_joints)
    loss_fn = MeanFlowLoss()
    
    joints = torch.randn(batch_size, n_joints)
    pose = torch.randn(batch_size, 7)
    
    loss, raw_loss = loss_fn(model, joints, pose)
    print(f"Loss: {loss.item()}, Raw Loss: {raw_loss.item()}")
    print("Loss function test passed!")
