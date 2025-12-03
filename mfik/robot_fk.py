"""
Robot forward kinematics using URDF files.
Provides RobotFK class for computing FK from URDF models.
"""

import torch
import sys
import os

# Add robots directory to path for urdf import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robots'))
from mfik.urdf import URDF


def rotation_matrix_to_quaternion(R):
    """
    将旋转矩阵转换为四元数 (x, y, z, w)
    
    Args:
        R (torch.Tensor): 形状为 (..., 3, 3) 的旋转矩阵
    
    Returns:
        torch.Tensor: 形状为 (..., 4) 的四元数 [x, y, z, w]
    """
    batch_shape = R.shape[:-2]
    R_flat = R.view(-1, 3, 3)
    batch_size = R_flat.shape[0]
    
    # 预分配四元数张量
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)
    
    # 计算迹
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    
    # 情况1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        q[mask1, 3] = 0.25 * s  # qw
        q[mask1, 0] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s  # qx
        q[mask1, 1] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s  # qy
        q[mask1, 2] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s  # qz
    
    # 情况2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2  # s = 4 * qx
        q[mask2, 3] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s  # qw
        q[mask2, 0] = 0.25 * s  # qx
        q[mask2, 1] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s  # qy
        q[mask2, 2] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s  # qz
    
    # 情况3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2  # s = 4 * qy
        q[mask3, 3] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s  # qw
        q[mask3, 0] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s  # qx
        q[mask3, 1] = 0.25 * s  # qy
        q[mask3, 2] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s  # qz
    
    # 情况4: 其他情况
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2  # s = 4 * qz
        q[mask4, 3] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s  # qw
        q[mask4, 0] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s  # qx
        q[mask4, 1] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s  # qy
        q[mask4, 2] = 0.25 * s  # qz
    
    return q.view(*batch_shape, 4)


class RobotFK:
    """
    Generic robot forward kinematics using URDF files.
    This class can load any robot from a URDF file and compute FK.
    """
    def __init__(self, urdf_path, device='cpu', end_link=None):
        """
        Initialize robot from URDF file.
        
        Args:
            urdf_path (str): Path to the URDF file
            device (str): 'cpu' or 'cuda'
            end_link (str): Name of the end effector link (default: last link in kinematic chain)
        """
        self.device = torch.device(device)
        self.urdf_path = urdf_path
        
        # Load URDF
        self.robot = URDF.load(urdf_path)
        self.actuated_joint_names = self.robot.actuated_joint_names
        self.n_joints = len(self.actuated_joint_names)
        
        # Determine end link
        if end_link is None:
            # Use the last link in the chain
            self.end_link = list(self.robot._G.nodes())[-1]
        else:
            self.end_link = end_link
            
        # Get joint limits
        joint_limits = self.robot.joint_limits
        q_min = []
        q_max = []
        for lower, upper in joint_limits:
            # Handle infinite limits
            if lower == float('-inf'):
                lower = -3.14159
            if upper == float('inf'):
                upper = 3.14159
            q_min.append(lower)
            q_max.append(upper)
            
        self.q_min = torch.tensor(q_min, device=self.device, dtype=torch.float32)
        self.q_max = torch.tensor(q_max, device=self.device, dtype=torch.float32)
        
    def forward_kinematics(self, q):
        """
        Computes FK for a batch of joint angles using the URDF model.
        
        Args:
            q: [Batch, n_joints] joint angles in radians
            
        Returns:
            pos: [Batch, 3] (x, y, z) positions
            quat: [Batch, 4] quaternions [qx, qy, qz, qw]
        """
        batch_size = q.shape[0]
        
        # Compute FK using URDF
        transforms = self.robot.link_fk_batch_gpu(
            cfgs=q, 
            device=self.device, 
            link=self.end_link,
            use_names=False
        )
        
        # Extract position
        pos = transforms[:, :3, 3]
        
        # Extract rotation and convert to quaternion
        rot_mat = transforms[:, :3, :3]
        quat = rotation_matrix_to_quaternion(rot_mat)
        
        return pos, quat
