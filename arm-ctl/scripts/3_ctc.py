"""arm-ctl/scripts/3_ctc.py — 计算力矩控制轨迹跟踪"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arm import ArmDynamics
from controllers import CTCController
from trajectories import CircleTrajectory, Figure8Trajectory
from simulate import run

dyn = ArmDynamics()
dt = 0.001
ctc = CTCController(dyn, omega_n=20.0, zeta=0.8)

circle = CircleTrajectory(period=4.0)
res_c = run(dyn, ctc, circle, duration=8.0, dt=dt)

fig8 = Figure8Trajectory(period=5.0)
res_f8 = run(dyn, ctc, fig8, duration=10.0, dt=dt)

print(f"CTC Circle Slow:  RMSE = {res_c['rmse']:.4f} rad")
print(f"CTC Figure-8:     RMSE = {res_f8['rmse']:.4f} rad")

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(res_c['times'], np.linalg.norm(res_c['e_hist'], axis=1), label='Circle')
ax1.plot(res_f8['times'], np.linalg.norm(res_f8['e_hist'], axis=1), label='Figure-8')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel(r'$\|e\|$ (rad)')
ax1.set_title('CTC Tracking Error')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot(res_c['pos_ref'][:, 0], res_c['pos_ref'][:, 1], res_c['pos_ref'][:, 2],
         'gray', ls='--', lw=1, label='Ref')
ax2.plot(res_c['pos_hist'][:, 0], res_c['pos_hist'][:, 1], res_c['pos_hist'][:, 2],
         'darkorange', lw=1, label='CTC')
ax2.set_title('Circle Path')
ax2.legend()

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot(res_f8['pos_ref'][:, 0], res_f8['pos_ref'][:, 1], res_f8['pos_ref'][:, 2],
         'gray', ls='--', lw=1, label='Ref')
ax3.plot(res_f8['pos_hist'][:, 0], res_f8['pos_hist'][:, 1], res_f8['pos_hist'][:, 2],
         'darkorange', lw=1, label='CTC')
ax3.set_title('Figure-8 Path')
ax3.legend()

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(res_c['times'], res_c['tau_hist'], lw=1)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Torque (Nm)')
ax4.set_title('CTC Torques (Circle)')
ax4.legend([r'$\tau_1$', r'$\tau_2$', r'$\tau_3$'])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('results/3_ctc.png', dpi=150)
plt.close()
print("Saved to results/3_ctc.png")
