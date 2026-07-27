# 3-DOF Robot Arm Trajectory Tracking

Three controllers (PID, Computed Torque Control, Neural Network Feedforward) driving a 3-DOF spatial robot arm to track 3D trajectories — comparing model-free, analytical model-based, and learned model-based approaches.

## Results

| Condition | PID (no model) | CTC (exact model) | NNFF (learned model) |
|-----------|---------------|-------------------|----------------------|
| Circle Slow (4s) | 0.0868 rad | **0.0012 rad** | 0.0030 rad |
| Circle Fast (3s) | 0.0927 rad | **0.0019 rad** | 0.0043 rad |
| Figure-8 (5s) | 0.0813 rad | **0.0013 rad** | 0.0031 rad |
| Figure-8 Fast (3s) | 0.0931 rad | **0.0027 rad** | 0.0039 rad |

CTC achieves ~0.1 degree tracking error on a 0.9m arm — equivalent to <2mm end-effector error.

## Quick Start

```bash
pip install -r requirements.txt

# Train the NN inverse dynamics model (required for NNFF)
python scripts/4_train_nn.py

# Run the comprehensive comparison
python scripts/6_compare.py

# Generate animations
python scripts/7_animate.py
python scripts/7b_compare_anim.py
```

## Project Structure

```
arm-ctl/
  src/
    arm.py            # Forward kinematics + dynamics (M, C, G, RK4)
    controllers.py    # PID, CTC, NNFeedforward controllers
    trajectories.py   # Circle/Figure-8 trajectories on tilted plane + IK
    viz.py            # 3D visualization, comparison plots, animations
    simulate.py       # Shared simulation runner
  scripts/            # Run sequentially: 1 → 2 → 3 → 4 → 5 → 6 → 7
  tests/              # Pytest test suite
  results/            # Generated plots and animations
```

## How It Works

### Arm Model
3-DOF anthropomorphic arm: base rotation (θ₁, about z) → shoulder pitch (θ₂) → elbow pitch (θ₃). Links 2 and 3 share the same rotation axis (no wrist joint). Dynamics derived from Euler-Lagrange with point-mass approximation:

```
M(q) ddq + C(q,dq) dq + G(q) = τ
```

### Controllers
- **PID** — Independent joint control, no dynamics model. Anti-windup.
- **CTC** — Feedback linearization using exact analytical M, C, G matrices.
- **NNFF** — Learned inverse dynamics (9→256→512→256→3 MLP) + PD feedback correction.

### Trajectories
Circle and figure-8 patterns defined on a 45° tilted plane, mapped to 3D via rotation.

## Tech Stack

Python 3.13+, NumPy, PyTorch, SciPy, Matplotlib

## License

MIT
