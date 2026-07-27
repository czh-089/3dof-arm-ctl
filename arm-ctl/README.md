# 3-DOF 机械臂轨迹跟踪控制：PID、计算力矩控制与神经网络前馈对比

## 摘要

本项目针对 3 自由度空间机械臂的轨迹跟踪问题，实现并对比了三种控制策略：独立关节 PID 控制（无模型）、计算力矩控制（CTC，基于解析动力学模型）和神经网络前馈控制（NNFF，基于数据驱动的逆动力学学习）。仿真在倾斜 45° 平面上以圆形和八字形轨迹进行测试，覆盖慢速和快速两种条件。结果表明：CTC 达到最优跟踪精度（RMSE < 0.003 rad，末端等效误差 < 2 mm），NNFF 次之（RMSE ~0.004 rad），PID 最差（RMSE ~0.09 rad）。

---

## 系统描述

### 机械臂构型

3-DOF 拟人臂：底座旋转（θ₁，绕 z 轴）→ 肩部俯仰（θ₂）→ 肘部俯仰（θ₃）。连杆 2 与连杆 3 共轴，无腕关节。臂在由 θ₁ 决定的竖直平面内运动。

物理参数：

| 参数 | l₁ | l₂ | l₃ | m₁ | m₂ | m₃ | g |
|------|----|----|----|----|----|----|---|
| 值 | 0.4 m | 0.3 m | 0.2 m | 2.0 kg | 1.5 kg | 1.0 kg | 9.81 m/s² |

### 动力学模型

采用 Euler-Lagrange 方法建立动力学方程，使用点质量近似：

$$M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + G(\mathbf{q}) = \boldsymbol{\tau}$$

其中 $M \in \mathbb{R}^{3\times 3}$ 为质量矩阵（$M = \sum m_i J_{v_i}^T J_{v_i}$），$C$ 由 Christoffel 符号通过有限差分计算，$G$ 为重力项。仿真采用 4 阶 Runge-Kutta 积分，步长 1 ms。

---

## 控制器设计

三种控制器共享统一接口：`compute_torque(t, q, dq, q_des, dq_des, ddq_des) → τ`。

### PID 控制（无模型 Baseline）

独立关节 PID + clamping anti-windup：

$$\tau_i = K_p e_i + K_d \dot{e}_i + K_i \int e_i \, dt$$

参数通过手动调试确定：$K_p = 100$, $K_d = 20$, $K_i = 10$。

### CTC（计算力矩控制）

基于反馈线性化，利用解析动力学矩阵完全抵消系统非线性：

$$\tau = M(\mathbf{q})\left(\ddot{\mathbf{q}}_{des} + K_p \mathbf{e} + K_d \dot{\mathbf{e}}\right) + C(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + G(\mathbf{q})$$

其中 $K_p = \omega_n^2$, $K_d = 2\zeta\omega_n$，取 $\omega_n = 20$, $\zeta = 0.8$。

### NNFF（神经网络前馈控制）

前馈网络学习逆动力学映射，PD 项修正残差：

$$\tau = f_{NN}(\mathbf{q}_{des}, \dot{\mathbf{q}}_{des}, \ddot{\mathbf{q}}_{des}) + K_p \mathbf{e} + K_d \dot{\mathbf{e}}$$

网络架构：9→256→512→256→3（ReLU 激活，Adam 优化器）。在 50,000 组随机采样的 $(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}})$ 上训练，标签由 `inverse_dynamics()` 生成。输入输出均经 z-score 标准化。

---

## 轨迹生成

两条轨迹定义在倾斜 45° 平面上（绕 x 轴旋转），通过逆运动学映射到关节空间：

- **圆形**：$u = R\cos(\omega t)$, $v = R\sin(\omega t)$，$R = 0.12$ m，中心 (0.3, 0.2, 0.3) m
- **八字形**：$u = a\sin(\omega t)$, $v = b\sin(2\omega t)$，$a = 0.12$ m, $b = 0.06$ m

逆运动学分两步求解：$\theta_1 = \text{atan2}(y, x)$ 确定臂平面，然后在平面内解双连杆几何。速度与加速度通过末端 Jacobian 传播。

---

## 实验结果

### 定量对比

4 种轨迹条件 × 3 种控制器 = 12 组实验。指标为关节空间 RMSE（单位：rad）：

| 轨迹条件 | PID | CTC | NNFF |
|----------|:---:|:---:|:----:|
| 圆形 (T=4s) | 0.087 | **0.0012** | 0.0030 |
| 圆形 (T=3s) | 0.093 | **0.0019** | 0.0043 |
| 八字形 (T=5s) | 0.081 | **0.0013** | 0.0031 |
| 八字形 (T=3s) | 0.093 | **0.0027** | 0.0039 |

### 分析

1. **CTC 在所有条件下表现最优**，跟踪误差接近数值积分精度。这验证了 Euler-Lagrange 动力学推导和反馈线性化实现的正确性。
2. **NN 前馈显著优于 PID**（提升约 20–30 倍），说明网络成功从数据中学习了逆动力学的近似映射，且能泛化到训练集外的轨迹条件。
3. **PID 在快速轨迹（T=3s）下误差增大**，反映无模型方法在高动态场景下受非线性耦合和重力影响较大。
4. 三种方法的性能排序与预期一致：CTC > NNFF ≫ PID，验证了引入模型信息对控制精度提升的价值。

---

## 项目结构

```
src/
  arm.py            # 正运动学、Jacobian、M/C/G 矩阵、RK4 积分
  controllers.py    # PID (anti-windup)、CTC、NN Feedforward
  trajectories.py   # 倾斜平面轨迹生成 + 逆运动学
  simulate.py       # 共享仿真运行器
  viz.py            # 3D 可视化、对比图、动画
scripts/            # 按编号顺序执行：被动 → PID → CTC → 训练 → NN 测试 → 对比 → 动画
tests/              # pytest 测试（17 个）
```

---

## 运行

```bash
pip install -r requirements.txt
python scripts/4_train_nn.py        # 训练逆动力学网络
python scripts/6_compare.py          # 12 组对比实验
python scripts/7b_compare_anim.py    # 并排动画
pytest tests/ -v                     # 单元测试
```

---

## 依赖

Python ≥ 3.10 · NumPy · PyTorch · SciPy · Matplotlib · Pillow

## License

MIT
