"""arm-ctl/scripts/7b_compare_anim.py — 三臂并排对比动画 GIF"""
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

sim_data = {}
for name, ctrl in controllers.items():
    reset_fn = ctrl.reset if hasattr(ctrl, 'reset') else None
    res = run(dyn, ctrl, circle, duration, dt, reset_fn=reset_fn)
    sim_data[name] = res
    print(f"{name}: RMSE = {res['rmse']:.4f} rad")

times_base = sim_data['PID']['times']
pos_ref = sim_data['PID']['pos_ref']

n_frames = 150
step = max(1, len(times_base) // n_frames)
frames = list(range(0, len(times_base), step))[:n_frames]

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                         subplot_kw={'projection': '3d'})
colors = {'PID': 'steelblue', 'CTC': 'darkorange', 'NNFF': 'seagreen'}

def update(idx):
    i = frames[idx]
    for ax, (name, data) in zip(axes, sim_data.items()):
        ax.clear()
        ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2],
                'gray', ls='--', lw=0.5, alpha=0.3)
        ax.plot(pos_ref[:i, 0], pos_ref[:i, 1], pos_ref[:i, 2],
                'gray', ls='--', lw=1, alpha=0.5)
        pts = dyn.fk.joint_positions(data['q_hist'][i])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o',
                color=colors[name], lw=3, markersize=5)
        ax.scatter(*pts[0], c='black', s=30)
        ee_done = [dyn.fk.joint_positions(data['q_hist'][j])[3]
                   for j in range(0, i, max(1, step))]
        if len(ee_done) > 1:
            ax.plot([p[0] for p in ee_done], [p[1] for p in ee_done],
                    [p[2] for p in ee_done], color=colors[name], lw=1.5, alpha=0.7)
        e_norm = np.linalg.norm(data['e_hist'][i])
        ax.set_title(f'{name}  |e|={e_norm:.3f} rad', fontsize=12, fontweight='bold')
        ax.set_xlim([-0.2, 0.7])
        ax.set_ylim([-0.3, 0.6])
        ax.set_zlim([-0.1, 0.9])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    fig.suptitle(f'PID vs CTC vs NNFF — Circle Tracking  t = {times_base[i]:.2f}s',
                 fontsize=14, fontweight='bold')

ani = FuncAnimation(fig, update, frames=len(frames),
                    interval=1000 / 25, blit=False)
ani.save('results/7_compare_anim.gif', writer='pillow', fps=25)
plt.close()
print("Saved results/7_compare_anim.gif")
