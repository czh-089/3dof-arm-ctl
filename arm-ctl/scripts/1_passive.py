"""arm-ctl/scripts/1_passive.py — 无控制自由摆动，验证动力学合理性"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arm import ArmDynamics
from simulate import run_passive
from viz import plot_arm_3d

dyn = ArmDynamics()
fk = dyn.fk

q0 = np.array([0.3, 0.6, -1.0])
dq0 = np.zeros(3)

res = run_passive(dyn, q0, dq0, duration=5.0, dt=0.001)
times, q_hist, dq_hist, energy_hist = (
    res['times'], res['q_hist'], res['dq_hist'], res['energy_hist'])

e_drift = (energy_hist[-1] - energy_hist[0]) / abs(energy_hist[0]) * 100

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
ax2.set_title(f'Energy drift: {e_drift:.4f}%')

ax3d = fig.add_subplot(2, 2, (3, 4), projection='3d')
n_steps = len(times)
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
assert abs(e_drift) < 0.5, f"Energy drift {e_drift:.3f}% too large!"
print("PASS: Energy conserved within tolerance")
print("Saved to results/1_passive.png")
