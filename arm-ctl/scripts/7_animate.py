"""arm-ctl/scripts/7_animate.py — 生成三控制器跟踪动画 GIF"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from arm import ArmDynamics
from controllers import PIDController, CTCController, NNFeedforward
from trajectories import CircleTrajectory, sample_trajectory


def simulate(dyn, controller, trajectory, duration, dt, reset_fn=None):
    times, q_ref, dq_ref, ddq_ref, pos_ref = sample_trajectory(
        trajectory, duration, dt)
    n_steps = len(times)
    q_hist = np.zeros((n_steps, 3))
    pos_hist = np.zeros((n_steps, 3))

    q = q_ref[0].copy()
    dq = np.zeros(3)
    if reset_fn:
        reset_fn()

    for i in range(n_steps):
        q_hist[i] = q
        tau = controller.compute_torque(times[i], q, dq,
                                        q_ref[i], dq_ref[i], ddq_ref[i])
        pos_hist[i] = dyn.fk.joint_positions(q)[3]
        q, dq = dyn.step(q, dq, tau, dt)

    return times, q_hist, pos_ref


def make_animation(times, q_hist, pos_ref, fk, title, filename, fps=25, n_frames=150):
    """生成单个控制器的跟踪动画"""
    step = max(1, len(times) // n_frames)
    frames = list(range(0, len(times), step))[:n_frames]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(idx):
        ax.clear()
        i = frames[idx]
        # 参考轨迹（已走过的部分）
        ax.plot(pos_ref[:i, 0], pos_ref[:i, 1], pos_ref[:i, 2],
                'gray', ls='--', lw=1, alpha=0.4)
        # 完整参考轨迹
        ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2],
                'gray', ls='--', lw=0.5, alpha=0.2)
        # 当前臂姿态
        pts = fk.joint_positions(q_hist[i])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o',
                color='steelblue', lw=3, markersize=6)
        ax.scatter(*pts[0], c='black', s=40)
        # 末端轨迹（已走过的）
        ee_done = np.array([fk.joint_positions(q_hist[j])[3] for j in range(0, i, max(1, step))])
        if len(ee_done) > 1:
            ax.plot(ee_done[:, 0], ee_done[:, 1], ee_done[:, 2],
                    'steelblue', lw=1.5, alpha=0.7)

        ax.set_xlim([-0.2, 0.7])
        ax.set_ylim([-0.3, 0.6])
        ax.set_zlim([-0.1, 0.9])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{title}  t = {times[i]:.2f}s')

    ani = FuncAnimation(fig, update, frames=len(frames),
                        interval=1000 / fps, blit=False)
    ani.save(filename, writer='pillow', fps=fps)
    plt.close()
    print(f"Saved {filename}")


if __name__ == '__main__':
    dyn = ArmDynamics()
    dt = 0.001

    # 用圆慢轨迹 (4s 周期), 跑 8s (2 个完整周期)
    circle = CircleTrajectory(period=4.0)
    duration = 8.0

    # PID
    pid = PIDController(kp=(100, 100, 100), kd=(20, 20, 20), ki=(10, 10, 10))
    pid.reset()
    times, q_hist_pid, pos_ref = simulate(dyn, pid, circle, duration, dt,
                                          reset_fn=pid.reset)
    make_animation(times, q_hist_pid, pos_ref, dyn.fk,
                   'PID Tracking', 'results/7_anim_pid.gif')

    # CTC
    ctc = CTCController(dyn, omega_n=20, zeta=0.8)
    _, q_hist_ctc, _ = simulate(dyn, ctc, circle, duration, dt)
    make_animation(times, q_hist_ctc, pos_ref, dyn.fk,
                   'CTC Tracking', 'results/7_anim_ctc.gif')

    # NNFF
    nnff = NNFeedforward(model_path='results/nn_inverse_dynamics.pt', kp=50, kd=10)
    _, q_hist_nnff, _ = simulate(dyn, nnff, circle, duration, dt)
    make_animation(times, q_hist_nnff, pos_ref, dyn.fk,
                   'NNFF Tracking', 'results/7_anim_nnff.gif')

    print("All animations generated.")
