"""arm-ctl/scripts/2_pid.py — PID 控制圆+八字轨迹跟踪"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arm import ArmDynamics, ForwardKinematics
from controllers import PIDController
from trajectories import (CircleTrajectory, Figure8Trajectory,
                          sample_trajectory)


def simulate_pid(dyn, controller, trajectory, duration, dt):
    """运行 PID 仿真，返回结果 dict"""
    times, q_ref, dq_ref, ddq_ref, pos_ref = sample_trajectory(
        trajectory, duration, dt)
    n_steps = len(times)

    q_hist = np.zeros((n_steps, 3))
    dq_hist = np.zeros((n_steps, 3))
    e_hist = np.zeros((n_steps, 3))
    tau_hist = np.zeros((n_steps, 3))
    pos_hist = np.zeros((n_steps, 3))

    controller.reset()
    q = q_ref[0].copy()
    dq = np.zeros(3)

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
    return {'times': times, 'q_hist': q_hist, 'dq_hist': dq_hist,
            'e_hist': e_hist, 'tau_hist': tau_hist,
            'pos_ref': pos_ref, 'q_ref': q_ref, 'pos_hist': pos_hist, 'rmse': rmse}


if __name__ == '__main__':
    dyn = ArmDynamics()
    dt = 0.001
    fk = dyn.fk

    # PID: 较高增益（无模型，靠反馈硬扛耦合和重力）
    pid = PIDController(kp=(100, 100, 100), kd=(20, 20, 20), ki=(10, 10, 10))

    # 圆慢 (4s 周期), 跑 2 个周期
    circle = CircleTrajectory(period=4.0)
    res_c = simulate_pid(dyn, pid, circle, duration=8.0, dt=dt)

    # 八字 (5s 周期), 跑 2 个周期
    fig8 = Figure8Trajectory(period=5.0)
    res_f8 = simulate_pid(dyn, pid, fig8, duration=10.0, dt=dt)

    print(f"PID Circle Slow:  RMSE = {res_c['rmse']:.4f} rad")
    print(f"PID Figure-8:     RMSE = {res_f8['rmse']:.4f} rad")

    # 绘制
    fig = plt.figure(figsize=(14, 10))

    # 误差曲线
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(res_c['times'], np.linalg.norm(res_c['e_hist'], axis=1), label='Circle')
    ax1.plot(res_f8['times'], np.linalg.norm(res_f8['e_hist'], axis=1), label='Figure-8')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'$\|e\|$ (rad)')
    ax1.set_title('PID Tracking Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 3D 路径 (圆)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot(res_c['pos_ref'][:, 0], res_c['pos_ref'][:, 1], res_c['pos_ref'][:, 2],
             'gray', ls='--', lw=1, label='Ref')
    ax2.plot(res_c['pos_hist'][:, 0], res_c['pos_hist'][:, 1], res_c['pos_hist'][:, 2],
             'steelblue', lw=1, label='PID')
    ax2.set_title('Circle End-Effector Path')
    ax2.legend()

    # 3D 路径 (八字)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot(res_f8['pos_ref'][:, 0], res_f8['pos_ref'][:, 1], res_f8['pos_ref'][:, 2],
             'gray', ls='--', lw=1, label='Ref')
    ax3.plot(res_f8['pos_hist'][:, 0], res_f8['pos_hist'][:, 1], res_f8['pos_hist'][:, 2],
             'darkorange', lw=1, label='PID')
    ax3.set_title('Figure-8 End-Effector Path')
    ax3.legend()

    # 力矩
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(res_c['times'], res_c['tau_hist'], lw=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Torque (Nm)')
    ax4.set_title('PID Control Torques (Circle)')
    ax4.legend([r'$\tau_1$', r'$\tau_2$', r'$\tau_3$'])
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('results/2_pid.png', dpi=150)
    plt.close()
    print("Saved to results/2_pid.png")
