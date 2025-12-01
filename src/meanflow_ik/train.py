import torch
import torch.optim as optim
import numpy as np
from torch.func import jvp, functional_call
from .model import MeanFlowNetwork
from .robot import PandaRobot
import time

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.robot = PandaRobot(device=self.device)
        
        # Model Setup
        # State dim = 7 (joints)
        # Condition dim = 7 (3 pos + 4 quat)
        self.model = MeanFlowNetwork(
            state_dim=7,
            condition_dim=7,
            hidden_dim=config.get('hidden_dim', 512),
            depth=config.get('depth', 6)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('lr', 1e-4))
        
        self.batch_size = config.get('batch_size', 256)
        self.iterations = config.get('iterations', 10000)
        self.mode = config.get('mode', 'meanflow') # 'baseline' or 'meanflow'

    def generate_batch(self):
        # 1. Sample q_gt
        q = torch.rand(self.batch_size, 7, device=self.device)
        q = q * (self.robot.q_max - self.robot.q_min) + self.robot.q_min
        
        # 2. Compute FK
        with torch.no_grad():
            pos, quat = self.robot.forward_kinematics(q)
            
        # Normalize q to [-1, 1]
        q_norm = 2 * (q - self.robot.q_min) / (self.robot.q_max - self.robot.q_min) - 1
        
        target_pose = torch.cat([pos, quat], dim=-1)
        
        return q_norm, target_pose

    def dispersive_loss(self, features, tau=0.5):
        # features: [B, D]
        # Compute pairwise distance squared
        dist_sq = torch.cdist(features, features, p=2) ** 2
        
        # exp(-dist/tau)
        kernel = torch.exp(-dist_sq / tau)
        
        # log(1/B * sum_i sum_j ...) = log(mean over i of sum over j)
        loss = torch.log(torch.mean(torch.sum(kernel, dim=1)))
        return loss

    def train_step_fm_baseline(self):
        self.optimizer.zero_grad()
        x0, condition = self.generate_batch()
        x1 = torch.randn_like(x0)
        t = torch.rand(self.batch_size, device=self.device)
        t_reshaped = t.view(-1, 1)
        
        z_t = (1 - t_reshaped) * x0 + t_reshaped * x1
        v_target = x1 - x0
        r = torch.zeros_like(t)
        
        v_pred, _ = self.model(z_t, r, t, condition)
        loss = torch.mean((v_pred - v_target) ** 2)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step_meanflow(self):
        self.optimizer.zero_grad()
        
        # Data (x0) and Noise (x1)
        x0, condition = self.generate_batch()
        x1 = torch.randn_like(x0)
        
        # Sample t, r ~ LogitNormal
        eps_t = torch.randn(self.batch_size, device=self.device)
        t = torch.sigmoid(eps_t)
        
        eps_r = torch.randn(self.batch_size, device=self.device)
        r = torch.sigmoid(eps_r)
        
        t_reshaped = t.view(-1, 1)
        
        # Interpolate z_t = (1-t)x0 + t*x1
        z_t = (1 - t_reshaped) * x0 + t_reshaped * x1
        v_t = x1 - x0 # dx/dt
        
        # JVP Calculation
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        
        def forward_closure(z, ref_t, curr_t, c):
             return functional_call(self.model, (params, buffers), (z, ref_t, curr_t, c))
             
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t), torch.zeros_like(condition))
        
        # jvp returns ((out, features), (out_dot, features_dot))
        (u_pred_no_grad, features_no_grad), (du_dt, _) = jvp(forward_closure, (z_t, r, t, condition), tangents)
        
        # u_target = v_t - (t-r) * du/dt
        tr_diff = (t - r).view(-1, 1)
        u_target = v_t - tr_diff * du_dt
        u_target = u_target.detach()
        
        # Prediction pass (with gradients)
        u_pred, features = self.model(z_t, r, t, condition)
        
        mse_loss = torch.mean((u_pred - u_target) ** 2)
        
        # Dispersive Loss
        disp_loss = 0.0
        if self.batch_size > 1:
            disp_loss = self.dispersive_loss(features, tau=0.5)
            
        lambda_disp = 0.25 
        
        loss = mse_loss + lambda_disp * disp_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def refine_solution(self, q_init, target_pose, steps=10, lr=0.05):
        q = q_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([q], lr=lr)
        
        target_pos = target_pose[:, :3]
        # target_quat = target_pose[:, 3:]
        
        for _ in range(steps):
            optimizer.zero_grad()
            pos, quat = self.robot.forward_kinematics(q)
            
            # Position error
            loss = torch.mean(torch.sum((pos - target_pos)**2, dim=-1))
            
            loss.backward()
            optimizer.step()
            
            # Clamp limits
            with torch.no_grad():
                q.data = torch.max(torch.min(q.data, self.robot.q_max), self.robot.q_min)
                
        return q.detach()

    def export_onnx(self, path="meanflow_ik.onnx"):
        self.model.eval()
        # Dummy inputs
        z = torch.randn(1, 7, device=self.device)
        r = torch.zeros(1, device=self.device)
        t = torch.ones(1, device=self.device)
        cond = torch.randn(1, 7, device=self.device)
        
        torch.onnx.export(
            self.model,
            (z, r, t, cond),
            path,
            input_names=["z", "r", "t", "condition"],
            output_names=["u", "features"],
            dynamic_axes={"z": {0: "batch"}, "condition": {0: "batch"}, "u": {0: "batch"}}
        )
        print(f"Model exported to {path}")

    def validate(self):
        print("Validating 1-NFE Inference...")
        self.model.eval()
        
        with torch.no_grad():
            q_gt = torch.rand(1000, 7, device=self.device)
            q_gt = q_gt * (self.robot.q_max - self.robot.q_min) + self.robot.q_min
            
            target_pos, target_quat = self.robot.forward_kinematics(q_gt)
            target_pose = torch.cat([target_pos, target_quat], dim=-1)
            
            start_time = time.time()
            q_pred = self.solve(target_pose)
            end_time = time.time()
            
            pred_pos, pred_quat = self.robot.forward_kinematics(q_pred)
            
            pos_err = torch.norm(pred_pos - target_pos, dim=-1)
            success_rate = (pos_err < 0.01).float().mean().item() * 100
            
            print(f"1-NFE Inference Speed: {(end_time - start_time)*1000/1000:.4f} ms/batch")
            print(f"1-NFE Mean Pos Error: {pos_err.mean().item()*100:.2f} cm")
            print(f"1-NFE Success Rate (<1cm): {success_rate:.2f}%")
            
            # Diversity Metric
            if batch_size > 1:
                # pdist requires 1D or flattened, but cdist works on batches
                # We want pairwise distance in joint space
                # q_pred: [B, 7]
                dists = torch.cdist(q_pred, q_pred, p=2)
                # Mean off-diagonal distance
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
                mean_dist = dists[mask].mean().item()
                print(f"1-NFE Diversity (Mean Joint Dist): {mean_dist:.4f}")
        
        # Refinement (requires grad)
        print("Validating Refined Inference...")
        start_time = time.time()
        q_refined = self.refine_solution(q_pred, target_pose, steps=20)
        end_time = time.time()
        
        with torch.no_grad():
            pred_pos_ref, _ = self.robot.forward_kinematics(q_refined)
            pos_err_ref = torch.norm(pred_pos_ref - target_pos, dim=-1)
            success_rate_ref = (pos_err_ref < 0.01).float().mean().item() * 100
            
            print(f"Refined Inference Speed: {(end_time - start_time)*1000/1000:.4f} ms/batch")
            print(f"Refined Mean Pos Error: {pos_err_ref.mean().item()*100:.2f} cm")
            print(f"Refined Success Rate (<1cm): {success_rate_ref:.2f}%")

    def train(self):
        print(f"Starting Training ({self.mode})...")
        for i in range(self.iterations):
            if self.mode == 'meanflow':
                loss = self.train_step_meanflow()
            else:
                loss = self.train_step_fm_baseline()
                
            if i % 100 == 0:
                print(f"Iter {i}: Loss = {loss:.6f}")
        
        self.validate()
        self.export_onnx()

    @torch.no_grad()
    def solve(self, target_pose):
        batch_size = target_pose.shape[0]
        z1 = torch.randn(batch_size, 7, device=self.device)
        
        r = torch.zeros(batch_size, device=self.device)
        t = torch.ones(batch_size, device=self.device)
        
        u, _ = self.model(z1, r, t, target_pose)
        
        z0_pred = z1 - u
        q_pred = (z0_pred + 1) * (self.robot.q_max - self.robot.q_min) / 2 + self.robot.q_min
        return q_pred
        
if __name__ == "__main__":
    config = {
        'hidden_dim': 256, 
        'depth': 4,
        'iterations': 1000,
        'mode': 'meanflow'
    }
    trainer = Trainer(config)
    trainer.train()