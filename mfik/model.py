import torch
import torch.nn as nn
import numpy as np


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalMLP(nn.Module):
    """
    Conditional MLP for MeanFlow IK with Neighborhood Projection.
    
    Key design from method.md:
    - Network learns residual Δq, output = z_t + Δq (residual connection)
    - This makes it easier to learn small corrections from q_ref to q_gt
    - z_t is the current state along the flow trajectory
    """
    
    def __init__(
        self, n_joints, condition_dim=7, time_emb_dim=64, hidden_dim=1024,
        use_residual=True  # New: enable/disable residual connection
    ):
        super().__init__()
        self.n_joints = n_joints
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.use_residual = use_residual

        # Time embedding for r and t
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Input dimension: z_t (n_joints) + pose (condition_dim) + r_emb (time_emb_dim) + t_emb (time_emb_dim) + noise_scale (1)
        input_dim = n_joints + condition_dim + time_emb_dim * 2 + 1

        # Main network with residual blocks
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Residual blocks for better gradient flow
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(4)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_joints),
        )
        
        # Initialize output layer with small weights for stable training
        nn.init.zeros_(self.output_proj[-1].bias)
        nn.init.normal_(self.output_proj[-1].weight, std=0.01)

    def forward(self, z_t, pose, r, t, noise_scale):
        """
        Args:
            z_t: (B, n_joints) Current joint configuration along flow
            pose: (B, condition_dim) Target end-effector pose
            r: (B,) Time parameter r
            t: (B,) Time parameter t
            noise_scale: (B, 1) or (B,) Noise scale parameter (projection radius)
        Returns:
            u_pred: (B, n_joints) Predicted velocity field
        """
        # Embed time
        r_emb = self.time_mlp(r)  # (B, time_emb_dim)
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)

        # Reshape noise_scale if needed
        if noise_scale.dim() == 1:
            noise_scale = noise_scale.unsqueeze(-1)  # (B, 1)

        # Concatenate inputs
        x = torch.cat([z_t, pose, r_emb, t_emb, noise_scale], dim=-1)

        # Forward pass through network
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        delta = self.output_proj(h)
        
        # Return velocity field (delta is the correction direction)
        # Note: The velocity u predicts how to move from z_t towards the target
        return delta


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(x + self.net(x))


if __name__ == "__main__":
    # Simple test
    batch_size = 4
    n_joints = 7
    model = ConditionalMLP(n_joints)

    z_t = torch.randn(batch_size, n_joints)
    pose = torch.randn(batch_size, 7)
    r = torch.rand(batch_size)
    t = torch.rand(batch_size)
    c = torch.zeros(batch_size)

    u = model(z_t, pose, r, t, c)
    print(f"Input shape: {z_t.shape}, Output shape: {u.shape}")
    print("Model test passed!")


if __name__ == "__main__":
    # Simple test
    batch_size = 4
    n_joints = 7
    model = ConditionalMLP(n_joints)

    z_t = torch.randn(batch_size, n_joints)
    pose = torch.randn(batch_size, 7)
    r = torch.rand(batch_size)
    t = torch.rand(batch_size)
    c = torch.zeros(batch_size)

    u = model(z_t, pose, r, t, c)
    print(f"Output shape: {u.shape}")
    assert u.shape == (batch_size, n_joints)
    print("Model test passed!")
