## ADDED Requirements

### Requirement: Real-time 1-NFE Inference
The IK solver SHALL generate joint configurations from end-effector poses using a single neural network function evaluation (1-NFE).

#### Scenario: Latency constraint
- **WHEN** a target pose is requested
- **THEN** the system returns a solution in approximately 1-5 milliseconds (on GPU)
- **AND** no iterative refinement is required for the base solution

### Requirement: High Accuracy Solutions
The generated joint configurations SHALL result in end-effector poses that match the target within a minimal tolerance.

#### Scenario: Positional accuracy
- **WHEN** a valid target pose is provided
- **THEN** the Forward Kinematics of the generated joints matches the target position
- **AND** the positional error is typically < 1cm (without refinement) or < 1mm (with optional refinement)

### Requirement: Diversity and Null-Space Coverage
The solver SHALL be capable of generating diverse solutions for redundant manipulators, covering the available null space.

#### Scenario: Mode seeking
- **WHEN** multiple batches of random noise are passed with the same target pose
- **THEN** the output joint configurations form distinct clusters or a continuous manifold in the null space
- **AND** the solutions are not collapsed to a single mode

### Requirement: Continuous Data Generation
The training pipeline SHALL generate training data procedurally on-the-fly to prevent overfitting.

#### Scenario: Infinite dataset
- **WHEN** the training loop requests a batch
- **THEN** random joint angles are sampled within limits
- **AND** corresponding Forward Kinematics are computed immediately
- **AND** data is normalized before being fed to the network

### Requirement: Dispersive Regularization
The training objective SHALL include a dispersive term to encourage feature-space separation of samples within a batch.

#### Scenario: Training loss
- **WHEN** calculating the loss for a batch
- **THEN** the pairwise distances between internal feature representations are computed
- **AND** a repulsive loss component is added to the total loss
