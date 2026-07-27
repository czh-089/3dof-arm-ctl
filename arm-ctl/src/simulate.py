"""arm-ctl/src/simulate.py — 共享仿真运行器，消除各脚本中的重复代码"""
import numpy as np
from trajectories import sample_trajectory


def run(dyn, controller, trajectory, duration, dt, reset_fn=None):
    """运行一次完整的轨迹跟踪仿真。

    Args:
        dyn: ArmDynamics 实例
        controller: 控制器对象 (有 compute_torque 方法)
        trajectory: Trajectory 实例
        duration: 仿真时长 (s)
        dt: 积分步长 (s)
        reset_fn: 可选, 控制器重置回调 (如 pid.reset)

    Returns dict with keys:
        times, q_hist, dq_hist, e_hist, tau_hist, pos_ref, q_ref, dq_ref,
        ddq_ref, pos_hist, rmse
    """
    times, q_ref, dq_ref, ddq_ref, pos_ref = sample_trajectory(
        trajectory, duration, dt)
    n_steps = len(times)

    q_hist = np.zeros((n_steps, 3))
    dq_hist = np.zeros((n_steps, 3))
    e_hist = np.zeros((n_steps, 3))
    tau_hist = np.zeros((n_steps, 3))
    pos_hist = np.zeros((n_steps, 3))

    q = q_ref[0].copy()
    dq = np.zeros(3)
    if reset_fn:
        reset_fn()

    for i in range(n_steps):
        q_hist[i] = q
        dq_hist[i] = dq
        e_hist[i] = q_ref[i] - q
        tau = controller.compute_torque(times[i], q, dq,
                                        q_ref[i], dq_ref[i], ddq_ref[i])
        tau_hist[i] = tau
        pos_hist[i] = dyn.fk.joint_positions(q)[3]
        q, dq = dyn.step(q, dq, tau, dt)

    rmse = np.sqrt(np.mean(np.sum(e_hist**2, axis=1)))
    return {
        'times': times,
        'q_hist': q_hist,
        'dq_hist': dq_hist,
        'e_hist': e_hist,
        'tau_hist': tau_hist,
        'pos_ref': pos_ref,
        'q_ref': q_ref,
        'dq_ref': dq_ref,
        'ddq_ref': ddq_ref,
        'pos_hist': pos_hist,
        'rmse': rmse,
    }


def run_passive(dyn, q0, dq0, duration, dt):
    """运行无力矩输入的自由摆动仿真。

    Returns dict with keys: times, q_hist, dq_hist, energy_hist
    """
    n_steps = int(duration / dt)
    times = np.arange(n_steps) * dt
    q_hist = np.zeros((n_steps, 3))
    dq_hist = np.zeros((n_steps, 3))
    energy_hist = np.zeros(n_steps)

    q, dq = q0.copy(), dq0.copy()
    for i in range(n_steps):
        q_hist[i] = q
        dq_hist[i] = dq
        M = dyn.mass_matrix(q)
        T = 0.5 * dq @ M @ dq
        coms = dyn.fk.com_positions(q)
        V = dyn.g * np.sum(dyn.masses * coms[:, 2])
        energy_hist[i] = T + V
        q, dq = dyn.step(q, dq, np.zeros(3), dt)

    return {
        'times': times,
        'q_hist': q_hist,
        'dq_hist': dq_hist,
        'energy_hist': energy_hist,
    }
