"""arm-ctl/scripts/1_passive.py — 无控制自由摆动，验证动力学合理性"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arm import ArmDynamics, ForwardKinematics
from viz import plot_arm_3d

dyn = ArmDynamics()
fk = dyn.fk

# 初始条件: 非零角度，零速度
q0 = np.array([0.3, 0.6, -1.0])
dq0 = np.zeros(3)

# 仿真 5 秒，dt=1ms
dt = 0.001
n_steps = int(5.0 / dt)
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
    coms = fk.com_positions(q)
    V = dyn.g * np.sum(dyn.masses * coms[:, 2])
    energy_hist[i] = T + V
    q, dq = dyn.step(q, dq, np.zeros(3), dt)

# 绘制
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(times, q_hist)
ax1.set_ylabel('Joint Angles (rad)')
ax1.legend([r'$\theta_1$', r'$\theta_2$', r'$\theta_3$'])
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(times, energy_hist)
ax2.set_ylabel('Total Energy (J)')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.3)
e_drift = (energy_hist[-1] - energy_hist[0]) / abs(energy_hist[0]) * 100
ax2.set_title(f'Energy drift: {e_drift:.4f}%')

# 3D 姿态快照
ax3d = fig.add_subplot(2, 2, (3, 4), projection='3d')
snapshots = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
for idx in snapshots:
    alpha = 0.3 + 0.7 * (idx / n_steps)
    plot_arm_3d(ax3d, q_hist[idx], fk, alpha=alpha, label=f't={times[idx]:.1f}s')
ax3d.set_xlim([-0.9, 0.9])
ax3d.set_ylim([-0.9, 0.9])
ax3d.set_zlim([-0.1, 0.9])
ax3d.set_xlabel('X (m)')
ax3d.set_ylabel('Y (m)')
ax3d.set_zlabel('Z (m)')
ax3d.legend(loc='upper right')

plt.tight_layout()
fig.savefig('results/1_passive.png', dpi=150)
plt.close()
print(f"Passive simulation done. Energy drift: {e_drift:.4f}%")
print(f"Final energy / Initial energy: {energy_hist[-1]:.4f} / {energy_hist[0]:.4f}")
assert abs(e_drift) < 0.5, f"Energy drift {e_drift:.3f}% too large!"
print("PASS: Energy conserved within tolerance")
print("Saved to results/1_passive.png")
