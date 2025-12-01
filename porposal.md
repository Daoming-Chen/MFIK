# Dispersive MeanFlow IK Solver - 系统设计文档 v1.0

## 1\. 项目概述 (Executive Summary)

### 1.1 背景

传统的数值 IK 求解器（如 TRAC-IK）在处理冗余机械臂（7-DoF+）时容易陷入局部极小值且无法提供多解。现有的生成式 IK（如 IKFlow）虽然解决了多解性问题，但受限于可逆神经网络（INN）的架构约束，且推理速度仍有优化空间。

### 1.2 目标

构建下一代 IK 求解器，实现以下核心指标：

  * **毫秒级推理 (Real-time):** 利用 MeanFlow 实现 **1-NFE (单次函数评估)** 的一步生成。
  * **高精度与全覆盖 (Accuracy & Diversity):** 利用 Dispersive Loss 防止模式坍缩，确保解空间覆盖机械臂的 Null Space。
  * **架构灵活 (Flexibility):** 摆脱 Normalizing Flow 的雅可比行列式限制，使用标准的 MLP 架构。

-----

## 2\. 核心数学模型 (Mathematical Formulation)

我们将 IK 问题建模为从**噪声分布** $\epsilon \sim \mathcal{N}(0, I)$ 到**关节空间解分布** $q \in \mathbb{R}^n$ 的流变换，条件是**末端位姿** $x_{target} \in SE(3)$。

### 2.1 MeanFlow 核心机制

不同于传统 Flow Matching 拟合瞬时速度 $v_t$，我们要拟合**平均速度** $u(z_t, r, t)$。
根据 MeanFlow 论文，训练目标是满足 **MeanFlow Identity**：
$$u(z_t, r, t) = v(z_t, t) - (t-r) \frac{d}{dt} u(z_t, r, t)$$

  * $z_t$: 时间 $t$ 时的插值状态。
  * $v(z_t, t)$: 瞬时速度（Ground Truth）。
  * $u(z_t, r, t)$: 神经网络预测的输出。

### 2.2 Dispersive Regularization

为了解决 IK 的多模态（Multi-modal）问题，我们在特征层引入斥力。
$$\mathcal{L}_{Disp} = \log \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{B} \exp \left( - \frac{\| h_i - h_j \|_2^2}{\tau} \right)$$

  * $h_i$: 网络中间层的特征向量。
  * $\tau$: 温度系数。
  * 该损失函数无需正样本对，仅利用 Batch 内的样本互斥性来最大化特征空间的利用率。

-----

## 3\. 系统架构 (System Architecture)

### 3.1 输入与输出定义

  * **输入 (Inputs):**
      * `target_pose`: 目标位姿向量 (位置 [x,y,z] + 旋转 [qx, qy, qz, qw] 或 6D 旋转表示)。维度: $7$ 或 $9$。
      * `z_t`: 当前带噪状态（混合了关节角度和噪声）。维度: $N_{dof}$ (如 Panda 为 7)。
      * `times`: 时间元组 $(t, r)$。
  * **条件嵌入 (Conditioning):**
      * 将 `target_pose` 与 时间编码 `PosEmb(t, t-r)` 拼接。
  * **网络骨干 (Backbone):**
      * **Residual MLP (ResMLP):** 由于 IK 数据维度低（相比图像），不需要 Transformer 或 CNN。
      * 结构：Input Layer -\> [ResBlock x 6] -\> Output Layer。
      * **ResBlock:** `Linear -> BatchNorm -> GELU -> Linear -> Dropout` + `Skip Connection`。
  * **输出 (Outputs):**
      * `u_pred`: 预测的平均速度向量。维度: $N_{dof}$。

### 3.2 损失函数模块

总损失函数为：
$$\mathcal{L}_{Total} = \mathcal{L}_{MeanFlow} + \lambda \cdot \mathcal{L}_{Disp}$$

1.  **$\mathcal{L}_{MeanFlow}$ (MSE Loss):**
    $$\| u_{\theta}(z_t, r, t, x_{target}) - \text{stop\_grad}(u_{target}) \|_2^2$$
    其中 $u_{target}$ 通过 JVP (Jacobian-Vector Product) 计算得到。

2.  **$\mathcal{L}_{Disp}$ (InfoNCE-like):**
    提取 ResMLP 第 3 或 第 4 个 Block 的激活值计算 Dispersive Loss。

-----

## 4\. 实现流程详解 (Implementation Details)

### 4.1 数据管线 (Data Pipeline)

IK 任务的优势在于数据可以无限生成。

1.  **关节采样:** 在机械臂关节限位 $[q_{min}, q_{max}]$ 内均匀采样 $q_{gt}$。
2.  **正运动学 (FK):** 计算 $x_{target} = FK(q_{gt})$。
3.  **标准化:**
      * 关节角度归一化到 $[-1, 1]$。
      * 笛卡尔位置归一化（减均值除方差）。
4.  **Batch 构建:** 每次训练实时生成 Batch $(q_{gt}, x_{target})$。

### 4.2 训练循环 (Training Loop - Pseudocode)

以下伪代码融合了 MeanFlow 的 JVP 训练技巧和 Dispersive Loss：

```python
# 伪代码 - 基于 PyTorch / JAX 逻辑

def train_step(batch, model, optimizer, lambda_disp):
    q_gt, x_target = batch
    B = q_gt.shape[0]

    # 1. 采样时间 t 和 r (使用 LogitNormal 分布)
    t = sample_logit_normal(B)
    r = sample_logit_normal(B)
    # 确保 t > r，并处理 t=r 的情况 (flow matching boundary)
    t, r = process_time_pairs(t, r)

    # 2. 构造流插值 (Flow Interpolation)
    eps = torch.randn_like(q_gt) # Prior noise
    z_t = (1 - t) * q_gt + t * eps
    
    # 瞬时速度 Ground Truth (Conditional Flow Matching)
    v_t = eps - q_gt 

    # 3. 定义前向函数用于 JVP 计算
    # 输入: z, r, t
    # 输出: u (预测速度), features (中间层特征用于 Dispersive Loss)
    def forward_fn(z, r_val, t_val):
        return model(z, r_val, t_val, condition=x_target)

    # 4. 执行 JVP (Jacobian-Vector Product)
    # 计算 du/dt。切线向量为 (v_t, 0, 1)，对应 (dz/dt, dr/dt, dt/dt)
    # torch.func.jvp 是实现这一步的关键，高效且只需一次反向传播
    (u_pred, features), (du_dt, _) = torch.func.jvp(
        forward_fn, 
        (z_t, r, t), 
        (v_t, torch.zeros_like(r), torch.ones_like(t))
    )

    # 5. 计算 MeanFlow Target
    # u_target = v_t - (t - r) * du/dt
    # 注意：这里必须使用 stop_gradient，避免二阶导数
    u_target = v_t - (t - r) * du_dt
    u_target = u_target.detach()

    # 6. 计算损失
    loss_mf = F.mse_loss(u_pred, u_target)
    
    # Dispersive Loss (应用于中间特征 features)
    # 使用 L2 距离的 InfoNCE 变体
    loss_disp = calculate_dispersive_loss(features, tau=0.5)

    total_loss = loss_mf + lambda_disp * loss_disp

    # 7. 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### 4.3 推理模式 (Inference / Generation)

由于 MeanFlow 训练的是从任意时刻 $t$ 到 $r$ 的平均速度，我们只需设 $t=1$ (噪声), $r=0$ (数据)，即可一步生成。

```python
def solve_ik(x_target, model, num_samples=100):
    # 1. 准备输入
    # x_target 复制 num_samples 份
    # eps 从标准正态分布采样
    eps = torch.randn(num_samples, dof)
    
    # 2. 设置时间
    t = torch.ones(num_samples, 1) # t=1 (Noise)
    r = torch.zeros(num_samples, 1) # r=0 (Data)

    # 3. 单次前向预测 (1-NFE)
    # 模型预测的是从 t 到 r 的平均速度 u
    with torch.no_grad():
        u_pred, _ = model(eps, r, t, condition=x_target)

    # 4. 欧拉积分一步到位
    # q = z_1 - (1 - 0) * u
    q_pred = eps - u_pred

    return denormalize(q_pred)
```

-----

## 5\. 关键超参数推荐 (Hyperparameters)

基于论文经验值，推荐以下初始设置：

| 参数 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| **Model Size** | Hidden Dim: 512, Depth: 6-8 | IK 问题维度低，不需要过深的网络 |
| **Activation** | GELU or Swish | 平滑的激活函数有助于梯度流 |
| **Sample Ratio** | $r \neq t$ 比例: 75% | 75% 的样本用于学习 MeanFlow，25% 用于学习边界 (Flow Matching) |
| **Lambda Disp ($\lambda$)** | 0.25 - 0.5 | 权重不宜过大，否则会破坏主任务的收敛 |
| **Temp Tau ($\tau$)** | 0.5 | Dispersive Loss 的温度系数 |
| **Time Sampling** | Logit-Normal ($\mu=0, \sigma=1$) | 相比均匀分布，更关注中间时刻的训练 |

-----

## 6\. 预期风险与解决方案

1.  **风险：JVP 计算开销**

      * **方案：** 确保使用 PyTorch 2.0+ 的 `torch.func` 或 JAX。JVP 实际上只增加约 20% 的训练开销，远小于二阶导数。

2.  **风险：精度不足 (毫米级误差)**

      * **方案：** 1. 增加 CFG (Classifier-Free Guidance)。在训练时以 10% 概率将 condition 置零。推理时使用 $u = u_{unc} + w(u_{cond} - u_{unc})$。
        2\. 将生成结果作为种子，输入到数值求解器（如 Levenberg-Marquardt）进行 1-2 步微调。由于种子非常好，微调只需 \<1ms。

3.  **风险：解的连续性**

      * **方案：** 对于连续轨迹任务，Dispersive Loss 可能会导致相邻帧跳变到不同的解模式。如果用于轨迹规划，建议在推理时引入上一帧的解作为额外的 condition，或者仅在初始点使用生成模型，后续使用数值微分。

## 7\. 开发路线图 (Roadmap)

1.  **Week 1: 基准复现**
      * 实现基础的 ResMLP 和 FK 数据生成器。
      * 实现标准的 Conditional Flow Matching (FM) 作为 Baseline。
2.  **Week 2: 引入 MeanFlow**
      * 实现 `sample_t_r` 和 JVP 训练逻辑。
      * 验证 1-NFE 生成能力，对比 Baseline 的推理速度。
3.  **Week 3: 引入 Dispersive Loss**
      * 在网络中间层加入 Dispersive Loss。
      * 使用 MMD (Maximum Mean Discrepancy) 指标评估解的多样性。
4.  **Week 4: 调优与部署**
      * 调整 $\lambda$ 和 $\tau$。
      * 导出为 ONNX/TensorRT，在实机上测试 Latency。