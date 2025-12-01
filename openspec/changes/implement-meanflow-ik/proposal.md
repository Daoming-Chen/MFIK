# Change: Implement Dispersive MeanFlow IK Solver

## Why
Traditional numerical IK solvers (e.g., TRAC-IK) struggle with redundant manipulators (7-DoF+), often getting stuck in local minima and failing to provide multiple solutions. Existing generative IK methods (e.g., IKFlow) are limited by Invertible Neural Network (INN) architecture constraints and suboptimal inference speed.

## What Changes
- Implement a new IK solver based on **MeanFlow** and **Dispersive Loss**.
- **ADDED** `ik-solver` capability, covering:
    - **Residual MLP Backbone**: Replaces constrained Normalizing Flows.
    - **MeanFlow Training**: Enables 1-NFE (Single Function Evaluation) generation.
    - **Dispersive Regularization**: Ensures solution diversity and null-space coverage.
    - **Infinite Data Pipeline**: Real-time FK-based data generation.

## Impact
- **Performance**: Achieves millisecond-level inference suitable for real-time control.
- **Quality**: Provides high-accuracy solutions with mode coverage (diversity).
- **Flexibility**: Decouples architecture from flow constraints, allowing standard deep learning optimizations.
