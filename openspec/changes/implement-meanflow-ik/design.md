# Design: Dispersive MeanFlow IK Solver

## Context
We are building a next-generation Inverse Kinematics (IK) solver for redundant manipulators. The goal is to overcome the speed limitations of diffusion models and the architectural constraints of normalizing flows (INNs), while maintaining high accuracy and solution diversity (handling the null space).

## Mathematical Formulation

We model the IK problem as a flow transformation from a **noise distribution** $\epsilon \sim \mathcal{N}(0, I)$ to the **joint space solution distribution** $q \in \mathbb{R}^n$, conditioned on the **end-effector pose** $x_{target} \in SE(3)$.

### MeanFlow Mechanism
Unlike traditional Flow Matching which fits instantaneous velocity $v_t$, MeanFlow fits the **average velocity** $u(z_t, r, t)$.
The training objective enforces the **MeanFlow Identity**:
$$u(z_t, r, t) = v(z_t, t) - (t-r) \frac{d}{dt} u(z_t, r, t)$$

- $z_t$: Interpolated state at time $t$.
- $v(z_t, t)$: Ground truth instantaneous velocity.
- $u(z_t, r, t)$: Network predicted output.

### Dispersive Regularization
To address multi-modality, we introduce a repulsive force in the feature space:
$$\mathcal{L}_{Disp} = \log \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{B} \exp \left( - \frac{\| h_i - h_j \|_2^2}{\tau} \right)$$

- $h_i$: Intermediate layer feature vector.
- $\tau$: Temperature coefficient.
- This unsupervised loss maximizes feature space utilization within a batch.

## System Architecture

### Inputs & Outputs
- **Inputs**:
  - `target_pose`: $[x, y, z]$ + $[qx, qy, qz, qw]$ (or 6D rotation). Dim: 7 or 9.
  - `z_t`: Noisy state (mixed joint angles). Dim: $N_{dof}$.
  - `times`: Tuple $(t, r)$.
- **Conditioning**: `target_pose` concatenated with Time Embeddings `PosEmb(t, t-r)`.
- **Backbone**: **Residual MLP (ResMLP)**
  - Structure: `Input -> [ResBlock x 6] -> Output`.
  - ResBlock: `Linear -> BatchNorm -> GELU -> Linear -> Dropout` + Skip Connection.
- **Output**: `u_pred` (Predicted average velocity, Dim: $N_{dof}$). 

### Loss Function
$$\mathcal{L}_{Total} = \mathcal{L}_{MeanFlow} + \lambda \cdot \mathcal{L}_{Disp}$$

1. **MeanFlow Loss (MSE)**: 
   Fit $u_{pred}$ to $u_{target} = v_t - (t - r) \cdot \frac{du}{dt}$.
   $\frac{du}{dt}$ is computed via Jacobian-Vector Product (JVP).
2. **Dispersive Loss**: 
   Applied to activations from the middle ResBlock.

## Implementation Details

### Data Pipeline
- **Infinite Generation**:
  1. Sample $q_{gt} \in [q_{min}, q_{max}]$.
  2. Compute $x_{target} = FK(q_{gt})$.
  3. Normalize inputs/outputs.
  4. Construct batch $(q_{gt}, x_{target})$ on the fly.

### Training Logic (JVP)
Key steps in the training loop:
1. Sample $t, r \sim \text{LogitNormal}$.
2. Construct $z_t = (1-t)q_{gt} + t\epsilon$.
3. Define `forward_fn(z, r, t)`.
4. Compute `(u_pred, features), (du_dt, _) = torch.func.jvp(forward_fn, (z_t, r, t), (v_t, 0, 1))`.
5. Compute `u_target = v_t - (t-r) * du_dt` (stop gradient).
6. Optimize $\mathcal{L}_{MF} + \lambda \mathcal{L}_{Disp}$.

### Inference (1-NFE)
For generation, we solve the flow from $t=1$ (noise) to $r=0$ (data) in a single step:
1. Sample $\epsilon \sim \mathcal{N}(0, I)$.
2. Set $t=1, r=0$.
3. Predict $u = \text{Model}(\epsilon, r, t, x_{target})$.
4. Integrate: $q_{pred} = \epsilon - u$.

## Decisions & Trade-offs

### Decision: MLP vs Transformer
- **Reasoning**: IK data is low-dimensional structured vectors, not sequences or images. Transformers add unnecessary overhead.
- **Choice**: Simple Residual MLP is sufficient and faster.

### Decision: JVP vs Hessian
- **Reasoning**: MeanFlow requires a time derivative. Calculating full Hessian is $O(N^2)$, but JVP is comparable to a forward pass $O(1)$ in terms of scaling.
- **Choice**: Use `torch.func.jvp` for efficient exact derivative computation.

### Risk: Precision
- **Mitigation**: If 1-NFE accuracy is insufficient (<1mm), use the prediction as a warm-start for a classic numerical solver (1-2 iterations max).

## Hyperparameters (Recommended)
- **Hidden Dim**: 512
- **Depth**: 6-8 layers
- **Lambda Disp**: 0.25 - 0.5
- **Tau**: 0.5
- **Time Sampling**: Logit-Normal ($\mu=0, \sigma=1$)
