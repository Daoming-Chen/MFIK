import torch
import torch.nn as nn
import math

class ResBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MeanFlowNetwork(nn.Module):
    def __init__(self, state_dim, condition_dim, hidden_dim=512, depth=6, dropout_rate=0.1, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input: state + condition + t_emb + (t-r)_emb
        input_total_dim = state_dim + condition_dim + 2 * time_emb_dim
        
        self.input_proj = nn.Linear(input_total_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout_rate) for _ in range(depth)
        ])

        self.output_proj = nn.Linear(hidden_dim, state_dim)
        
        self.feature_layer_idx = depth // 2  # Extract from middle

    def forward(self, z, r, t, condition):
        """
        z: Noisy state [Batch, D]
        r: Reference time [Batch] (or scalar)
        t: Current time [Batch] (or scalar)
        condition: Conditioning vector (e.g., target pose) [Batch, C]
        """
        
        # Expand scalar times if necessary
        if isinstance(r, (int, float)):
            r = torch.full((z.shape[0],), r, device=z.device)
        if isinstance(t, (int, float)):
            t = torch.full((z.shape[0],), t, device=z.device)

        t_emb = self.time_mlp(t)
        tr_emb = self.time_mlp(t - r)
        
        # Concatenate all inputs
        # [z, condition, t_emb, tr_emb]
        x = torch.cat([z, condition, t_emb, tr_emb], dim=-1)
        
        x = self.input_proj(x)

        features = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Store features for dispersive loss (Phase 3)
            if i == self.feature_layer_idx:
                features = x

        out = self.output_proj(x)
        return out, features

    # I will rewrite the class to properly handle inputs as described.
