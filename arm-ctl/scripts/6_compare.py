"""arm-ctl/scripts/6_compare.py — PID vs CTC vs NNFF 综合对比"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arm import ArmDynamics
from controllers import PIDController, CTCController, NNFeedforward
from trajectories import (CircleTrajectory, Figure8Trajectory,
                          sample_trajectory)
from viz import plot_comparison


def run_trial(dyn, controller, trajectory, duration, dt, reset_fn=None):
    times, q_ref, dq_ref, ddq_ref, pos_ref = sample_trajectory(
        trajectory, duration, dt)
    n_steps = len(times)
    e_hist = np.zeros((n_steps, 3))
    tau_hist = np.zeros((n_steps, 3))
    pos_hist = np.zeros((n_steps, 3))

    q = q_ref[0].copy()
    dq = np.zeros(3)
    if reset_fn:
        reset_fn()

    for i in range(n_steps):
        e_hist[i] = q_ref[i] - q
        tau = controller.compute_torque(times[i], q, dq,
                                        q_ref[i], dq_ref[i], ddq_ref[i])
        tau_hist[i] = tau
        pos_hist[i] = dyn.fk.joint_positions(q)[3]
        q, dq = dyn.step(q, dq, tau, dt)

    return {'times': times, 'e_history': e_hist, 'tau_history': tau_hist,
            'pos_ref': pos_ref, 'pos_actual': pos_hist}


if __name__ == '__main__':
    dyn = ArmDynamics()
    dt = 0.001

    pid = PIDController(kp=(100, 100, 100), kd=(20, 20, 20), ki=(10, 10, 10))
    ctc = CTCController(dyn, omega_n=20, zeta=0.8)
    nnff = NNFeedforward(model_path='results/nn_inverse_dynamics.pt', kp=50, kd=10)

    conditions = [
        ("Circle Slow", CircleTrajectory(period=4.0), 8.0),
        ("Circle Fast", CircleTrajectory(period=3.0), 6.0),
        ("Figure-8", Figure8Trajectory(period=5.0), 10.0),
        ("Figure-8 Fast", Figure8Trajectory(period=3.0), 6.0),
    ]

    summary_rmse = []

    for cond_name, traj, duration in conditions:
        print(f"\n=== {cond_name} ===")
        cond_results = {}

        pid.reset()
        cond_results['PID'] = run_trial(dyn, pid, traj, duration, dt,
                                        reset_fn=pid.reset)
        cond_results['CTC'] = run_trial(dyn, ctc, traj, duration, dt)
        cond_results['NNFF'] = run_trial(dyn, nnff, traj, duration, dt)

        for name, res in cond_results.items():
            rmse = np.sqrt(np.mean(np.sum(res['e_history']**2, axis=1)))
            print(f"  {name:6s}: RMSE = {rmse:.4f} rad")
            summary_rmse.append((cond_name, name, rmse))

        # 每个条件生成对比图
        safe_name = cond_name.replace(' ', '_').replace('-', '').lower()
        plot_comparison(cond_results,
                        save_path=f'results/6_compare_{safe_name}.png')

    # 汇总表格
    print("\n=== RMSE Summary ===")
    print(f"{'Condition':<16} {'PID':>10} {'CTC':>10} {'NNFF':>10}")
    print("-" * 48)
    cond_names = [c[0] for c in conditions]
    for cn in cond_names:
        pid_r = [r for c, n, r in summary_rmse if c == cn and n == 'PID'][0]
        ctc_r = [r for c, n, r in summary_rmse if c == cn and n == 'CTC'][0]
        nnff_r = [r for c, n, r in summary_rmse if c == cn and n == 'NNFF'][0]
        print(f"{cn:<16} {pid_r:10.4f} {ctc_r:10.4f} {nnff_r:10.4f}")

    # 汇总柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cond_names))
    width = 0.25
    colors = ['steelblue', 'darkorange', 'seagreen']
    for i, (name, color) in enumerate([('PID', colors[0]),
                                        ('CTC', colors[1]),
                                        ('NNFF', colors[2])]):
        vals = [r for c, n, r in summary_rmse if n == name]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(cond_names)
    ax.set_ylabel('RMSE (rad)')
    ax.set_title('PID vs CTC vs NNFF — Tracking Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig('results/6_compare_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved all comparison plots to results/")
