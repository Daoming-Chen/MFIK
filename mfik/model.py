import torch
import torch.nn as nn
import math

class ResBlockAdaLN(nn.Module):
    def __init__(self, dim, cond_dim, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # adaLN modulation: input condition, output scale/shift
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * dim, bias=True)
        )
        
        # Zero initialization for adaLN (identity mapping at start)
        with torch.no_grad():
            self.adaLN_modulation[1].weight.zero_()
            self.adaLN_modulation[1].bias.zero_()

    def forward(self, x, c):
        shift_scale = self.adaLN_modulation(c)
        shift1, scale1, shift2, scale2 = shift_scale.chunk(4, dim=1)
        
        # Block 1
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        h = self.act(self.linear1(h))
        
        # Block 2
        h = self.norm2(h)
        h = h * (1 + scale2) + shift2
        h = self.dropout(self.linear2(h))
        
        return x + h

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
        # We keep the input projection as is (early fusion), but also use adaLN
        input_total_dim = state_dim + condition_dim + 2 * time_emb_dim
        
        # Dimension of the conditioning vector for adaLN
        # c = [condition, t_emb, tr_emb]
        self.cond_dim = condition_dim + 2 * time_emb_dim
        
        self.input_proj = nn.Linear(input_total_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResBlockAdaLN(hidden_dim, self.cond_dim, dropout_rate) for _ in range(depth)
        ])

        self.output_proj = nn.Linear(hidden_dim, state_dim)
        
        self.feature_layer_idx = depth // 2  # Extract from middle
        
        # Initialize output projection to zero (standard practice for flow matching)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

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
        
        # Construct conditioning vector c
        c = torch.cat([condition, t_emb, tr_emb], dim=-1)
        
        # Input projection (Early Fusion)
        # [z, c]
        x_input = torch.cat([z, c], dim=-1)
        x = self.input_proj(x_input)

        features = None
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            # Store features for dispersive loss (Phase 3)
            if i == self.feature_layer_idx:
                features = x

        out = self.output_proj(x)
        return out, features

    # I will rewrite the class to properly handle inputs as described.
