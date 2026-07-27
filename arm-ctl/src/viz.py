"""arm-ctl/src/viz.py — 3D 机械臂可视化"""
import numpy as np
import matplotlib.pyplot as plt


def plot_arm_3d(ax, q, fk, color='steelblue', alpha=0.8, label=None):
    """在 3D 轴上绘制机械臂连杆"""
    pts = fk.joint_positions(q)
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    ax.plot(xs, ys, zs, '-o', color=color, lw=3, markersize=6,
            alpha=alpha, label=label)
    ax.scatter(*pts[0], c='black', s=40, zorder=5)


def animate_tracking(times, q_history, q_ref_history, pos_ref_history, fk,
                     trajectory, filename='arm_tracking.gif', fps=30):
    """生成 3D 跟踪动画 GIF"""
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 确定固定视角范围
    all_pos = np.concatenate([pos_ref_history, pos_ref_history], axis=0)
    x_range = [all_pos[:, 0].min() - 0.3, all_pos[:, 0].max() + 0.3]
    y_range = [all_pos[:, 1].min() - 0.3, all_pos[:, 1].max() + 0.3]
    z_range = [-0.1, 0.9]

    def update(frame):
        ax.clear()
        ax.plot(pos_ref_history[:, 0], pos_ref_history[:, 1], pos_ref_history[:, 2],
                'gray', ls='--', lw=1, alpha=0.5)
        plot_arm_3d(ax, q_ref_history[frame], fk, color='gray', alpha=0.3, label='Desired')
        plot_arm_3d(ax, q_history[frame], fk, color='steelblue', alpha=0.8, label='Actual')
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(loc='upper left')
        ax.set_title(f't = {times[frame]:.2f}s')

    step = max(1, len(times) // 200)
    frames = list(range(0, len(times), step))
    ani = FuncAnimation(fig, update, frames=frames,
                        interval=1000 / fps, blit=False)
    ani.save(filename, writer='pillow', fps=fps)
    plt.close()
    print(f"Animation saved to {filename}")


def plot_comparison(results_dict, save_path='results/comparison.png'):
    """绘制多控制器对比图

    results_dict: {name: {'times': t, 'e_history': e_hist, 'tau_history': tau_hist,
                          'pos_ref': pos, 'pos_actual': pos}, ...}
    """
    fig = plt.figure(figsize=(14, 10))

    # 左上: 跟踪误差范数曲线
    ax1 = fig.add_subplot(2, 2, 1)
    for name, data in results_dict.items():
        e_norm = np.linalg.norm(data['e_history'], axis=1)
        ax1.plot(data['times'], e_norm, lw=1.5, label=name)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'$\|e\|$ (rad)')
    ax1.set_title('Tracking Error Norm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右上: RMSE 柱状图
    ax2 = fig.add_subplot(2, 2, 2)
    names = list(results_dict.keys())
    rmse_vals = [np.sqrt(np.mean(np.sum(d['e_history']**2, axis=1)))
                 for d in results_dict.values()]
    colors = ['steelblue', 'darkorange', 'seagreen'][:len(names)]
    bars = ax2.bar(names, rmse_vals, color=colors)
    for bar, val in zip(bars, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('RMSE (rad)')
    ax2.set_title('Joint Space RMSE')

    # 左下: 关节 1 力矩对比
    ax3 = fig.add_subplot(2, 2, 3)
    for name, data in results_dict.items():
        ax3.plot(data['times'], data['tau_history'][:, 0], lw=1, label=name)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(r'$\tau_1$ (Nm)')
    ax3.set_title('Joint 1 Torque')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 右下: 3D 跟踪路径
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    first_key = list(results_dict.keys())[0]
    ref = results_dict[first_key].get('pos_ref')
    if ref is not None:
        ax4.plot(ref[:, 0], ref[:, 1], ref[:, 2], 'gray', ls='--', lw=1, label='Ref')
    for (name, data), c in zip(results_dict.items(), colors):
        pos = data.get('pos_actual')
        if pos is not None:
            ax4.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=1, color=c, label=name)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('End-Effector Path')
    ax4.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison saved to {save_path}")
