"""arm-ctl/scripts/7_animate.py — 生成三控制器跟踪动画 GIF"""
import sys
sys.path.insert(0, 'src')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from arm import ArmDynamics
from controllers import PIDController, CTCController, NNFeedforward
from trajectories import CircleTrajectory
from simulate import run

dyn = ArmDynamics()
dt = 0.001
circle = CircleTrajectory(period=4.0)
duration = 8.0

controllers = {
    'PID': PIDController(kp=(100, 100, 100), kd=(20, 20, 20), ki=(10, 10, 10)),
    'CTC': CTCController(dyn, omega_n=20, zeta=0.8),
    'NNFF': NNFeedforward(model_path='results/nn_inverse_dynamics.pt', kp=50, kd=10),
}

n_frames = 150

for name, ctrl in controllers.items():
    reset_fn = ctrl.reset if hasattr(ctrl, 'reset') else None
    res = run(dyn, ctrl, circle, duration, dt, reset_fn=reset_fn)
    times, q_hist, pos_ref = res['times'], res['q_hist'], res['pos_ref']

    step = max(1, len(times) // n_frames)
    frames = list(range(0, len(times), step))[:n_frames]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(idx, q_hist=q_hist, pos_ref=pos_ref, times=times):
        ax.clear()
        i = frames[idx]
        ax.plot(pos_ref[:i, 0], pos_ref[:i, 1], pos_ref[:i, 2],
                'gray', ls='--', lw=1, alpha=0.4)
        ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2],
                'gray', ls='--', lw=0.5, alpha=0.2)
        pts = dyn.fk.joint_positions(q_hist[i])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o',
                color='steelblue', lw=3, markersize=6)
        ax.scatter(*pts[0], c='black', s=40)
        ee_done = [dyn.fk.joint_positions(q_hist[j])[3]
                   for j in range(0, i, max(1, step))]
        if len(ee_done) > 1:
            ax.plot([p[0] for p in ee_done], [p[1] for p in ee_done],
                    [p[2] for p in ee_done], 'steelblue', lw=1.5, alpha=0.7)
        ax.set_xlim([-0.2, 0.7])
        ax.set_ylim([-0.3, 0.6])
        ax.set_zlim([-0.1, 0.9])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{name} Tracking  t = {times[i]:.2f}s')

    ani = FuncAnimation(fig, update, frames=len(frames),
                        interval=1000 / 25, blit=False)
    filename = f'results/7_anim_{name.lower()}.gif'
    ani.save(filename, writer='pillow', fps=25)
    plt.close()
    print(f"Saved {filename}")
