"""arm-ctl/scripts/2_pid.py — PID 控制圆+八字轨迹跟踪"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arm import ArmDynamics
from controllers import PIDController
from trajectories import CircleTrajectory, Figure8Trajectory
from simulate import run

dyn = ArmDynamics()
dt = 0.001

pid = PIDController(kp=(100, 100, 100), kd=(20, 20, 20), ki=(10, 10, 10))

circle = CircleTrajectory(period=4.0)
res_c = run(dyn, pid, circle, duration=8.0, dt=dt, reset_fn=pid.reset)

fig8 = Figure8Trajectory(period=5.0)
res_f8 = run(dyn, pid, fig8, duration=10.0, dt=dt, reset_fn=pid.reset)

print(f"PID Circle Slow:  RMSE = {res_c['rmse']:.4f} rad")
print(f"PID Figure-8:     RMSE = {res_f8['rmse']:.4f} rad")

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(res_c['times'], np.linalg.norm(res_c['e_hist'], axis=1), label='Circle')
ax1.plot(res_f8['times'], np.linalg.norm(res_f8['e_hist'], axis=1), label='Figure-8')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel(r'$\|e\|$ (rad)')
ax1.set_title('PID Tracking Error')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot(res_c['pos_ref'][:, 0], res_c['pos_ref'][:, 1], res_c['pos_ref'][:, 2],
         'gray', ls='--', lw=1, label='Ref')
ax2.plot(res_c['pos_hist'][:, 0], res_c['pos_hist'][:, 1], res_c['pos_hist'][:, 2],
         'steelblue', lw=1, label='PID')
ax2.set_title('Circle End-Effector Path')
ax2.legend()

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot(res_f8['pos_ref'][:, 0], res_f8['pos_ref'][:, 1], res_f8['pos_ref'][:, 2],
         'gray', ls='--', lw=1, label='Ref')
ax3.plot(res_f8['pos_hist'][:, 0], res_f8['pos_hist'][:, 1], res_f8['pos_hist'][:, 2],
         'darkorange', lw=1, label='PID')
ax3.set_title('Figure-8 End-Effector Path')
ax3.legend()

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
