## 1. Phase 1: Baseline & Foundation
- [x] 1.1 Implement basic Residual MLP (ResMLP) backbone with skip connections
- [x] 1.2 Implement Forward Kinematics (FK) data generator for the target robot (e.g., Panda)
- [x] 1.3 Implement standard Conditional Flow Matching (FM) training loop as a baseline
- [x] 1.4 Verify baseline convergence and basic generation capability

## 2. Phase 2: MeanFlow Core
- [x] 2.1 Implement `sample_t_r` logic using Logit-Normal distribution
- [x] 2.2 Implement `forward_fn` and JVP-based loss calculation using `torch.func.jvp`
- [x] 2.3 Implement MeanFlow loss function (MSE between predicted and target average velocity)
- [x] 2.4 Implement 1-NFE inference logic (step from t=1 noise to r=0 data)
- [x] 2.5 Validate 1-NFE generation quality and speed against baseline

## 3. Phase 3: Dispersive Regularization
- [x] 3.1 Implement feature extraction hook in ResMLP (e.g., at block 3 or 4)
- [x] 3.2 Implement Dispersive Loss (InfoNCE-like pairwise repulsion)
- [x] 3.3 Integrate Dispersive Loss into the main training loop with weight $\lambda$
- [x] 3.4 Implement evaluation metrics: Success Rate and Diversity (e.g., pairwise distance, MMD)

## 4. Phase 4: Optimization & Deployment
- [x] 4.1 Tune hyperparameters: Network depth, $\lambda$, Dispersive temperature $\tau$
- [x] 4.2 Implement optional numerical refinement (e.g., 1-2 steps of Newton-Raphson) for high-precision requirements
- [x] 4.3 Export trained model to ONNX/TensorRT formats
- [x] 4.4 Conduct final latency and accuracy benchmarks
