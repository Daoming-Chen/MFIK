import torch

class PandaRobot:
    def __init__(self, device='cpu'):
        self.device = device
        # DH Parameters for Franka Emika Panda
        # a, d, alpha, theta_offset
        self.dh_params = torch.tensor([
            [0.0,    0.333, 0.0,         0.0],
            [0.0,    0.0,   -1.570796327, 0.0],
            [0.0,    0.316, 1.570796327,  0.0],
            [0.0825, 0.0,   1.570796327,  0.0],
            [-0.0825, 0.384, -1.570796327, 0.0],
            [0.0,    0.0,   1.570796327,  0.0],
            [0.088,  0.0,   1.570796327,  0.0] 
        ], device=device)
        # Flange transformation is usually added at the end or as a fixed tool frame.
        # Here we consider the 7th link frame as the end for simplicity or add an offset.
        # The standard Panda FK usually goes to the flange.
        # Let's add the flange offset as a fixed transformation if needed, 
        # but for 7-DoF IK, solving for the 7th frame (or flange) is the goal.
        
        # Joint limits (approximate)
        self.q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], device=device)
        self.q_max = torch.tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], device=device)

    def forward_kinematics(self, q):
        """
        Computes FK for a batch of joint angles.
        q: [Batch, 7]
        Returns:
            pos: [Batch, 3] (x, y, z)
            rot: [Batch, 3, 3] rotation matrices or quaternion? 
                 Proposal says: target_pose: [x, y, z] + [qx, qy, qz, qw]
        """
        batch_size = q.shape[0]
        
        # Identity matrix for base
        T = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        for i in range(7):
            a = self.dh_params[i, 0]
            d = self.dh_params[i, 1]
            alpha = self.dh_params[i, 2]
            offset = self.dh_params[i, 3]
            theta = q[:, i] + offset
            
            # Transformation matrix for current joint
            # T_i = [cos(th), -sin(th)cos(alpha),  sin(th)sin(alpha), a*cos(th)]
            #       [sin(th),  cos(th)cos(alpha), -cos(th)sin(alpha), a*sin(th)]
            #       [0,        sin(alpha),         cos(alpha),        d]
            #       [0,        0,                  0,                 1]
            
            ct = torch.cos(theta)
            st = torch.sin(theta)
            ca = torch.cos(alpha)
            sa = torch.sin(alpha)
            
            # Expand scalars to batch size
            ca = ca.expand_as(ct)
            sa = sa.expand_as(ct)
            d_exp = d.expand_as(ct)
            a_exp = a.expand_as(ct)
            zeros = torch.zeros_like(ct)
            
            row0 = torch.stack([ct, -st*ca, st*sa, a_exp*ct], dim=1)
            row1 = torch.stack([st, ct*ca, -ct*sa, a_exp*st], dim=1)
            row2 = torch.stack([zeros, sa, ca, d_exp], dim=1)
            row3 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(batch_size, 1)
            
            Ti = torch.stack([row0, row1, row2, row3], dim=1)
            
            T = torch.bmm(T, Ti)
            
        # Extract position
        pos = T[:, :3, 3]
        rot_mat = T[:, :3, :3]
        
        # Convert rotation matrix to quaternion
        quat = self.matrix_to_quaternion(rot_mat)
        
        return pos, quat

    def matrix_to_quaternion(self, matrix):
        """
        Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw].
        """
        # Simple implementation or use a library if complex. 
        # For robustness, let's implement a standard conversion.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
        # But torch doesn't have built-in mat2quat in core without pytorch3d.
        # I'll implement a robust one.
        
        # Based on standard algorithm
        trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
        
        # This is a simplified version, might need branching for stability.
        # For now, assume trace > 0 for simplicity or implement branching.
        # Let's implement a slightly more robust one with branching (often used in graphics).
        
        # Actually, for minimal implementation and batching, a trace-check implementation is good.
        
        batch_size = matrix.shape[0]
        q = torch.zeros(batch_size, 4, device=self.device)
        
        # We'll use a simplified logic for now, assuming well-behaved matrices.
        # Or better: use a trace-based selection.
        
        # ... (Implementation details)
        # Let's use a safe implementation.
        
        tr = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
        
        # Check trace > 0
        mask_pos = tr > 0
        
        # Case 1: tr > 0
        S = torch.sqrt(tr[mask_pos] + 1.0) * 2
        q[mask_pos, 3] = 0.25 * S
        q[mask_pos, 0] = (matrix[mask_pos, 2, 1] - matrix[mask_pos, 1, 2]) / S
        q[mask_pos, 1] = (matrix[mask_pos, 0, 2] - matrix[mask_pos, 2, 0]) / S
        q[mask_pos, 2] = (matrix[mask_pos, 1, 0] - matrix[mask_pos, 0, 1]) / S
        
        # Case 2: tr <= 0 (simplified, taking max diagonal)
        # This is tedious to implement fully robustly in one go without helper.
        # I will stick to the trace > 0 case for the prototype if acceptable, 
        # but IK covers full space, so rotations can be anything.
        # I should handle at least the major cases.
        
        # Actually, let's just assume I implement a placeholder or a simple one 
        # and refine if needed. Or better, since we are in Python, I can copy a robust implementation.
        
        return self._robust_mat2quat(matrix)

    def _robust_mat2quat(self, matrix):
        # A robust implementation (e.g. from multiple sources)
        # [w, x, y, z] convention or [x, y, z, w]?
        # Proposal says [qx, qy, qz, qw]
        
        m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
        m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
        m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
        
        trace = m00 + m11 + m22
        
        q = torch.zeros(matrix.shape[0], 4, device=matrix.device)
        
        # Case 1: trace > 0
        mask = trace > 0
        s = torch.sqrt(trace[mask] + 1.0) * 2
        q[mask, 3] = 0.25 * s
        q[mask, 0] = (m21[mask] - m12[mask]) / s
        q[mask, 1] = (m02[mask] - m20[mask]) / s
        q[mask, 2] = (m10[mask] - m01[mask]) / s
        
        # Case 2: m00 is largest
        mask = ~mask & (m00 > m11) & (m00 > m22)
        s = torch.sqrt(1.0 + m00[mask] - m11[mask] - m22[mask]) * 2
        q[mask, 3] = (m21[mask] - m12[mask]) / s
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (m01[mask] + m10[mask]) / s
        q[mask, 2] = (m02[mask] + m20[mask]) / s
        
        # Case 3: m11 is largest
        # We need to update the mask to exclude already handled ones
        # Re-evaluate remaining
        remaining = (q[:, 3] == 0) & (q[:, 0] == 0) & (q[:, 1] == 0) & (q[:, 2] == 0) # Not perfect check but okay since we fill
        # Better: just use exclusive masks.
        
        # Let's restart mask logic clearly.
        cond1 = trace > 0
        cond2 = (~cond1) & (m00 > m11) & (m00 > m22)
        cond3 = (~cond1) & (~cond2) & (m11 > m22)
        cond4 = (~cond1) & (~cond2) & (~cond3)
        
        # 1
        if cond1.any():
            s = torch.sqrt(trace[cond1] + 1.0) * 2
            q[cond1, 3] = 0.25 * s
            q[cond1, 0] = (m21[cond1] - m12[cond1]) / s
            q[cond1, 1] = (m02[cond1] - m20[cond1]) / s
            q[cond1, 2] = (m10[cond1] - m01[cond1]) / s
            
        # 2
        if cond2.any():
            s = torch.sqrt(1.0 + m00[cond2] - m11[cond2] - m22[cond2]) * 2
            q[cond2, 3] = (m21[cond2] - m12[cond2]) / s
            q[cond2, 0] = 0.25 * s
            q[cond2, 1] = (m01[cond2] + m10[cond2]) / s
            q[cond2, 2] = (m02[cond2] + m20[cond2]) / s
            
        # 3
        if cond3.any():
            s = torch.sqrt(1.0 + m11[cond3] - m00[cond3] - m22[cond3]) * 2
            q[cond3, 3] = (m02[cond3] - m20[cond3]) / s
            q[cond3, 0] = (m01[cond3] + m10[cond3]) / s
            q[cond3, 1] = 0.25 * s
            q[cond3, 2] = (m12[cond3] + m21[cond3]) / s
            
        # 4
        if cond4.any():
            s = torch.sqrt(1.0 + m22[cond4] - m00[cond4] - m11[cond4]) * 2
            q[cond4, 3] = (m10[cond4] - m01[cond4]) / s
            q[cond4, 0] = (m02[cond4] + m20[cond4]) / s
            q[cond4, 1] = (m12[cond4] + m21[cond4]) / s
            q[cond4, 2] = 0.25 * s
            
        # Return [x, y, z, w]
        return q
