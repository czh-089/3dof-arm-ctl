"""arm-ctl/src/controllers.py — PID / CTC / NNFF 控制器"""
import numpy as np
import torch
import torch.nn as nn


class PIDController:
    """独立关节 PID + anti-windup (clamping)

    统一接口: compute_torque(t, q, dq, q_des, dq_des, ddq_des) → tau (3,)
    """

    def __init__(self, kp=(80, 80, 80), kd=(15, 15, 15), ki=(5, 5, 5),
                 tau_max=(50, 50, 50), dt=0.001):
        self.kp = np.array(kp, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.tau_max = np.array(tau_max, dtype=float)
        self.dt = dt
        self.integral = np.zeros(3)

    def reset(self):
        self.integral = np.zeros(3)

    def compute_torque(self, t, q, dq, q_des, dq_des, ddq_des):
        e = q_des - q
        de = dq_des - dq

        # Clamping anti-windup: 只在当前积分项不导致饱和时才积分
        tau_unsaturated = self.kp * e + self.kd * de + self.ki * self.integral
        unsaturated = np.abs(tau_unsaturated) < self.tau_max
        self.integral[unsaturated] += e[unsaturated] * self.dt

        tau_raw = self.kp * e + self.kd * de + self.ki * self.integral
        tau = np.clip(tau_raw, -self.tau_max, self.tau_max)
        return tau


class CTCController:
    """计算力矩控制: τ = M(q)(ddq_des + Kp e + Kd e_dot) + C(q,dq)dq + G(q)

    用解析动力学做反馈线性化。增益由 ω_n, ζ 参数化:
      Kp = ω_n², Kd = 2ζω_n
    """

    def __init__(self, dynamics, omega_n=20.0, zeta=0.8):
        self.dyn = dynamics
        self.omega_n = omega_n
        self.zeta = zeta
        self.kp = omega_n ** 2
        self.kd = 2 * zeta * omega_n

    def compute_torque(self, t, q, dq, q_des, dq_des, ddq_des):
        e = q_des - q
        de = dq_des - dq
        ddq_ref = ddq_des + self.kp * e + self.kd * de
        M = self.dyn.mass_matrix(q)
        c = self.dyn.coriolis_torque(q, dq)
        G = self.dyn.gravity_vector(q)
        return M @ ddq_ref + c + G


class InverseDynamicsNet(nn.Module):
    """9→256→512→256→3 逆动力学网络
    输入: [q(3), dq(3), ddq(3)] = 9维
    输出: τ(3)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.net(x)


class NNFeedforward:
    """NN 前馈 + PD 反馈: τ = NN(q, dq, ddq_des) + Kp·e + Kd·ė

    NN 学习逆动力学，PD 修正残差。输入使用参考轨迹 (q_des, dq_des, ddq_des)
    作为前馈——这是标准的前馈+反馈架构。
    """

    def __init__(self, model_path=None, kp=50.0, kd=10.0):
        self.kp = kp
        self.kd = kd
        self.model = InverseDynamicsNet()
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self._device = 'cpu'
        if model_path:
            self.load(model_path)
        self.model.eval()

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.X_mean = checkpoint['X_mean']
        self.X_std = checkpoint['X_std']
        self.Y_mean = checkpoint['Y_mean']
        self.Y_std = checkpoint['Y_std']
        # 获取模型参数所在设备
        self._device = next(self.model.parameters()).device

    def compute_torque(self, t, q, dq, q_des, dq_des, ddq_des):
        e = q_des - q
        de = dq_des - dq

        x = np.concatenate([q_des, dq_des, ddq_des])
        if self.X_mean is not None:
            x = (x - self.X_mean) / self.X_std
        x_t = torch.tensor(x, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            y_norm = self.model(x_t).cpu().numpy()
        tau_ff = y_norm * self.Y_std + self.Y_mean if self.Y_mean is not None else y_norm

        tau_pd = self.kp * e + self.kd * de
        return tau_ff + tau_pd
