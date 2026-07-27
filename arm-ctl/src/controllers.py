"""arm-ctl/src/controllers.py — PID / CTC / NNFF 控制器"""
import numpy as np


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
