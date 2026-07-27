"""arm-ctl/src/trajectories.py — 轨迹生成与逆运动学"""
import numpy as np


def inverse_kinematics(target, l1=0.4, l2=0.3, l3=0.2, elbow_up=True):
    """3-DOF 臂逆运动学：给定末端位置 (x,y,z)，返回关节角 (θ₁,θ₂,θ₃)

    步骤：
    1. θ₁ = atan2(y, x) — 旋转臂平面到目标方向
    2. 在臂平面内解 2 连杆 IK (l1, l2+l3):
       r = sqrt(x² + y²), z_target = z
       cos(θ₃) = (R² - l1² - leff²) / (2*l1*leff)
       θ₂ = atan2(z, r) - atan2(leff*sin(θ₃), l1+leff*cos(θ₃))
    """
    x, y, z = target
    r = np.sqrt(x**2 + y**2)
    th1 = np.arctan2(y, x) if r > 1e-10 else 0.0

    leff = l2 + l3
    R = np.sqrt(r**2 + z**2)
    R = np.clip(R, abs(l1 - leff) + 1e-6, l1 + leff - 1e-6)

    cos_th3 = (R**2 - l1**2 - leff**2) / (2 * l1 * leff)
    cos_th3 = np.clip(cos_th3, -1.0, 1.0)
    th3 = np.arccos(cos_th3)
    if not elbow_up:
        th3 = -th3

    alpha = np.arctan2(leff * np.sin(th3), l1 + leff * np.cos(th3))
    th2 = np.arctan2(z, r) - alpha

    return np.array([th1, th2, th3])


def ik_jacobian(q, l1=0.4, l2=0.3, l3=0.2):
    """末端位置关于关节角的 3×3 Jacobian（解析）"""
    th1, th2, th3 = q
    c1, s1 = np.cos(th1), np.sin(th1)
    c2, s2 = np.cos(th2), np.sin(th2)
    c23, s23 = np.cos(th2 + th3), np.sin(th2 + th3)
    leff = l2 + l3

    r = l1 * c2 + leff * c23

    J = np.zeros((3, 3))
    J[0, 0] = -r * s1
    J[1, 0] = r * c1
    J[2, 0] = 0.0
    J[0, 1] = (-l1 * s2 - leff * s23) * c1
    J[1, 1] = (-l1 * s2 - leff * s23) * s1
    J[2, 1] = l1 * c2 + leff * c23
    J[0, 2] = (-leff * s23) * c1
    J[1, 2] = (-leff * s23) * s1
    J[2, 2] = leff * c23

    return J


class Trajectory:
    """轨迹基类 — 定义在倾斜平面上的笛卡尔轨迹

    平面坐标系 (u, v)：u 水平，v 沿倾斜方向。
    通过绕 x 轴旋转 -tilt_angle 映射到 3D 空间。
    """

    def __init__(self, center=(0.3, 0.2, 0.3), tilt_angle=np.pi/4):
        self.center = np.array(center, dtype=float)
        ca, sa = np.cos(tilt_angle), np.sin(tilt_angle)
        self.R_plane_to_3d = np.array([
            [1.0, 0.0, 0.0],
            [0.0, ca, -sa],
            [0.0, sa, ca],
        ])

    def planar_pos(self, t):
        raise NotImplementedError

    def planar_vel(self, t):
        raise NotImplementedError

    def planar_acc(self, t):
        raise NotImplementedError

    def position(self, t):
        u, v = self.planar_pos(t)
        return self.center + self.R_plane_to_3d @ np.array([u, v, 0.0])

    def velocity(self, t):
        du, dv = self.planar_vel(t)
        return self.R_plane_to_3d @ np.array([du, dv, 0.0])

    def acceleration(self, t):
        ddu, ddv = self.planar_acc(t)
        return self.R_plane_to_3d @ np.array([ddu, ddv, 0.0])


class CircleTrajectory(Trajectory):
    """倾斜平面上的圆轨迹: u = R*cos(ωt), v = R*sin(ωt)"""

    def __init__(self, radius=0.12, period=4.0, center=(0.3, 0.2, 0.3), tilt_angle=np.pi/4):
        super().__init__(center, tilt_angle)
        self.radius = radius
        self.omega = 2 * np.pi / period
        self._period = period  # for test scripts

    def planar_pos(self, t):
        th = self.omega * t
        return np.array([self.radius * np.cos(th), self.radius * np.sin(th)])

    def planar_vel(self, t):
        th = self.omega * t
        return np.array([-self.radius * self.omega * np.sin(th),
                         self.radius * self.omega * np.cos(th)])

    def planar_acc(self, t):
        th = self.omega * t
        return np.array([-self.radius * self.omega**2 * np.cos(th),
                         -self.radius * self.omega**2 * np.sin(th)])


class Figure8Trajectory(Trajectory):
    """倾斜平面上的八字形 (Lissajous): u = a*sin(ωt), v = b*sin(2ωt)"""

    def __init__(self, a=0.12, b=0.06, period=5.0, center=(0.3, 0.2, 0.3), tilt_angle=np.pi/4):
        super().__init__(center, tilt_angle)
        self.a, self.b = a, b
        self.omega = 2 * np.pi / period
        self._period = period

    def planar_pos(self, t):
        wt = self.omega * t
        return np.array([self.a * np.sin(wt), self.b * np.sin(2 * wt)])

    def planar_vel(self, t):
        wt = self.omega * t
        return np.array([self.a * self.omega * np.cos(wt),
                         2 * self.b * self.omega * np.cos(2 * wt)])

    def planar_acc(self, t):
        wt = self.omega * t
        return np.array([-self.a * self.omega**2 * np.sin(wt),
                         -4 * self.b * self.omega**2 * np.sin(2 * wt)])


def sample_trajectory(trajectory, duration, dt, l1=0.4, l2=0.3, l3=0.2):
    """返回 (times, q_ref, dq_ref, ddq_ref, pos_ref)，各 (N,3) 或 (N,)"""
    n_steps = int(duration / dt) + 1
    times = np.arange(n_steps) * dt

    q_ref = np.zeros((n_steps, 3))
    dq_ref = np.zeros((n_steps, 3))
    ddq_ref = np.zeros((n_steps, 3))
    pos_ref = np.zeros((n_steps, 3))

    for i, t in enumerate(times):
        pos_ref[i] = trajectory.position(t)
        vel_3d = trajectory.velocity(t)
        acc_3d = trajectory.acceleration(t)

        q_ref[i] = inverse_kinematics(pos_ref[i], l1, l2, l3)

        J = ik_jacobian(q_ref[i], l1, l2, l3)
        dq_ref[i] = np.linalg.solve(J, vel_3d)

        eps = 1e-6
        q_plus = q_ref[i] + eps * dq_ref[i]
        J_plus = ik_jacobian(q_plus, l1, l2, l3)
        J_dot = (J_plus - J) / eps
        ddq_ref[i] = np.linalg.solve(J, acc_3d - J_dot @ dq_ref[i])

    return times, q_ref, dq_ref, ddq_ref, pos_ref
