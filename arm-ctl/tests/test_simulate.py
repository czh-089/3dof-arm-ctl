"""测试共享仿真运行器"""
import numpy as np
from simulate import run, run_passive
from controllers import PIDController
from trajectories import CircleTrajectory


def test_run_returns_expected_keys(dyn):
    pid = PIDController()
    circle = CircleTrajectory()
    res = run(dyn, pid, circle, duration=0.05, dt=0.001, reset_fn=pid.reset)
    expected_keys = {'times', 'q_hist', 'dq_hist', 'e_hist', 'tau_hist',
                     'pos_ref', 'q_ref', 'dq_ref', 'ddq_ref', 'pos_hist', 'rmse'}
    assert expected_keys <= set(res.keys()), f"Missing keys: {expected_keys - set(res.keys())}"


def test_run_tracking_rmse_reasonable(dyn):
    """PID 控制器跟踪误差应在合理范围内"""
    pid = PIDController()
    circle = CircleTrajectory()
    res = run(dyn, pid, circle, duration=0.5, dt=0.001, reset_fn=pid.reset)
    # 从零误差开始，误差会增长但应保持在合理范围
    assert res['rmse'] < 0.3, f"PID RMSE {res['rmse']:.4f} too large"


def test_run_passive_energy(dyn):
    q0 = np.array([0.3, 0.6, -1.0])
    res = run_passive(dyn, q0, np.zeros(3), duration=0.5, dt=0.001)
    e = res['energy_hist']
    drift = abs(e[-1] - e[0]) / abs(e[0])
    assert drift < 0.01, f"Energy drift {drift*100:.3f}%"
