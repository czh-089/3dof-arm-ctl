"""测试逆运动学与轨迹生成"""
import numpy as np
from trajectories import (inverse_kinematics, ik_jacobian,
                          CircleTrajectory, Figure8Trajectory,
                          sample_trajectory)


def test_ik_fk_closure_circle(fk):
    circle = CircleTrajectory()
    errors = []
    for t in np.linspace(0, circle._period, 30):
        pos_3d = circle.position(t)
        q = inverse_kinematics(pos_3d)
        ee = fk.joint_positions(q)[3]
        errors.append(np.linalg.norm(ee - pos_3d))
    assert max(errors) < 0.01, f"Max IK error {max(errors):.2e} too large"


def test_ik_fk_closure_figure8(fk):
    fig8 = Figure8Trajectory()
    errors = []
    for t in np.linspace(0, fig8._period, 30):
        pos_3d = fig8.position(t)
        q = inverse_kinematics(pos_3d)
        ee = fk.joint_positions(q)[3]
        errors.append(np.linalg.norm(ee - pos_3d))
    assert max(errors) < 0.01, f"Max IK error {max(errors):.2e} too large"


def test_ik_at_origin():
    """测试 target 在 z 轴正上方时 IK 不崩溃"""
    q = inverse_kinematics((0.0, 0.0, 0.5))
    assert q.shape == (3,)


def test_jacobian_velocity_propagation():
    """测试 J * dq 给出末端速度"""
    q = np.array([0.3, 0.6, -1.0])
    dq = np.array([0.1, 0.2, -0.15])
    J = ik_jacobian(q)
    v_ee = J @ dq
    assert v_ee.shape == (3,)
    assert np.linalg.norm(v_ee) > 0


def test_sample_trajectory():
    circle = CircleTrajectory()
    times, q_ref, dq_ref, ddq_ref, pos_ref = sample_trajectory(
        circle, duration=1.0, dt=0.01)
    assert len(times) == 101
    assert q_ref.shape == (101, 3)
    assert dq_ref.shape == (101, 3)
    assert ddq_ref.shape == (101, 3)
    assert pos_ref.shape == (101, 3)
