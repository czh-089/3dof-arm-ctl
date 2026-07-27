"""验证 IK 闭环：给定轨迹 → IK → FK 末端位置 → 应与轨迹一致"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from arm import ForwardKinematics
from trajectories import (CircleTrajectory, Figure8Trajectory,
                          inverse_kinematics, sample_trajectory)

fk = ForwardKinematics()
circle = CircleTrajectory()
fig8 = Figure8Trajectory()

for name, traj in [("Circle", circle), ("Figure8", fig8)]:
    errors = []
    for t in np.linspace(0, traj._period, 50):
        pos_3d = traj.position(t)
        q = inverse_kinematics(pos_3d)
        ee = fk.joint_positions(q)[3]
        err = np.linalg.norm(ee - pos_3d)
        errors.append(err)
    max_err = max(errors)
    print(f"{name}: max IK error = {max_err:.2e} m")
    assert max_err < 0.01, f"{name} IK error too large!"

# Test sample_trajectory
print("\nTesting sample_trajectory...")
times, q_ref, dq_ref, ddq_ref, pos_ref = sample_trajectory(circle, duration=2.0, dt=0.01)
print(f"  Sampled {len(times)} points over {times[-1]:.1f}s")
print(f"  q_ref range: [{q_ref.min():.2f}, {q_ref.max():.2f}]")
print(f"  dq_ref max: {np.abs(dq_ref).max():.2f} rad/s")

# Verify FK closure at sampled points
ee_errors = []
for i in range(len(times)):
    ee = fk.joint_positions(q_ref[i])[3]
    ee_errors.append(np.linalg.norm(ee - pos_ref[i]))
print(f"  Max EE error in sampled trajectory: {max(ee_errors):.2e} m")
assert max(ee_errors) < 0.01

print("PASS: IK 验证通过")
