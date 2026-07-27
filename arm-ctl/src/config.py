"""arm-ctl/src/config.py — 集中参数配置

所有物理参数、控制器增益、轨迹参数在此定义，各模块从此导入。
"""

# --- 机械臂物理参数 ---
ARM_PARAMS = {
    "l1": 0.4,   # 连杆 1 长度 (m)
    "l2": 0.3,   # 连杆 2 长度 (m)
    "l3": 0.2,   # 连杆 3 长度 (m)
    "m1": 2.0,   # 连杆 1 质量 (kg)
    "m2": 1.5,   # 连杆 2 质量 (kg)
    "m3": 1.0,   # 连杆 3 质量 (kg)
    "g":  9.81,  # 重力加速度 (m/s²)
}

# --- 仿真参数 ---
SIM_PARAMS = {
    "dt": 0.001,        # 积分步长 (s)
}

# --- PID 控制器参数 ---
PID_PARAMS = {
    "kp": (100, 100, 100),
    "kd": (20, 20, 20),
    "ki": (10, 10, 10),
    "tau_max": (50, 50, 50),
}

# --- CTC 控制器参数 ---
CTC_PARAMS = {
    "omega_n": 20.0,    # 固有频率 (rad/s)
    "zeta": 0.8,        # 阻尼比
}

# --- NN 前馈参数 ---
NN_PARAMS = {
    "kp": 50.0,
    "kd": 10.0,
    "hidden": [256, 512, 256],
    "lr": 1e-3,
    "batch_size": 512,
    "n_epochs": 200,
    "n_samples": 50000,
}

# --- 轨迹参数 ---
CIRCLE_SLOW = {"radius": 0.12, "period": 4.0, "center": (0.3, 0.2, 0.3)}
CIRCLE_FAST = {"radius": 0.12, "period": 3.0, "center": (0.3, 0.2, 0.3)}
FIGURE8 = {"a": 0.12, "b": 0.06, "period": 5.0, "center": (0.3, 0.2, 0.3)}
FIGURE8_FAST = {"a": 0.12, "b": 0.06, "period": 3.0, "center": (0.3, 0.2, 0.3)}
