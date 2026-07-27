"""测试 PID 和 CTC 控制器基本行为"""
import numpy as np
from controllers import PIDController, CTCController


def test_pid_produces_torque_with_error():
    pid = PIDController()
    tau = pid.compute_torque(0, np.zeros(3), np.zeros(3),
                             np.array([0.5, 0.3, -0.8]),
                             np.zeros(3), np.zeros(3))
    assert np.any(np.abs(tau) > 0)


def test_pid_zero_at_setpoint_after_reset():
    pid = PIDController()
    pid.reset()
    tau = pid.compute_torque(0, np.ones(3), np.zeros(3),
                             np.ones(3), np.zeros(3), np.zeros(3))
    assert np.allclose(tau, 0, atol=1e-10)


def test_ctc_compensates_gravity(dyn):
    ctc = CTCController(dyn)
    q = np.zeros(3)
    tau = ctc.compute_torque(0, q, np.zeros(3), q, np.zeros(3), np.zeros(3))
    G = dyn.gravity_vector(q)
    assert np.allclose(tau, G, atol=0.1)


def test_ctc_error_produces_larger_torque(dyn):
    ctc = CTCController(dyn)
    q = np.zeros(3)
    G = dyn.gravity_vector(q)
    tau = ctc.compute_torque(0, q, np.zeros(3),
                             np.array([0.5, 0.3, -0.8]),
                             np.zeros(3), np.zeros(3))
    assert np.any(np.abs(tau) > np.abs(G))
