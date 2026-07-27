"""arm-ctl/src/arm.py — 3-DOF 拟人臂正向运动学与动力学"""
import numpy as np


class ForwardKinematics:
    """3-DOF 拟人臂正向运动学（COM 位置 + Jacobians）

    构型: 底座旋转(θ₁,绕z) → 肩部俯仰(θ₂,臂平面内) → 肘部俯仰(θ₃,同轴)
    连杆3与连杆2同向（无腕关节），臂在由θ₁决定的竖直平面内运动。
    """

    def __init__(self, l1=0.4, l2=0.3, l3=0.2):
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.a1, self.a2, self.a3 = l1 / 2, l2 / 2, l3 / 2

    def com_positions(self, q):
        """返回 (3, 3) 数组：每行是一个连杆 COM 的 (x, y, z)"""
        th1, th2, th3 = q
        c1, s1 = np.cos(th1), np.sin(th1)
        c2, s2 = np.cos(th2), np.sin(th2)
        c23, s23 = np.cos(th2 + th3), np.sin(th2 + th3)

        r1 = self.a1 * c2
        z1 = self.a1 * s2
        r2 = self.l1 * c2 + self.a2 * c23
        z2 = self.l1 * s2 + self.a2 * s23
        leff = self.l2 + self.a3
        r3 = self.l1 * c2 + leff * c23
        z3 = self.l1 * s2 + leff * s23

        return np.array([
            [r1 * c1, r1 * s1, z1],
            [r2 * c1, r2 * s1, z2],
            [r3 * c1, r3 * s1, z3],
        ])

    def joint_positions(self, q):
        """返回 (4, 3) 数组：base → joint2 → joint3 → end-effector"""
        th1, th2, th3 = q
        c1, s1 = np.cos(th1), np.sin(th1)
        c2, s2 = np.cos(th2), np.sin(th2)
        c23, s23 = np.cos(th2 + th3), np.sin(th2 + th3)

        r_j2 = self.l1 * c2
        z_j2 = self.l1 * s2
        r_j3 = r_j2 + self.l2 * c23
        z_j3 = z_j2 + self.l2 * s23
        r_ee = r_j3 + self.l3 * c23
        z_ee = z_j3 + self.l3 * s23

        return np.array([
            [0.0, 0.0, 0.0],
            [r_j2 * c1, r_j2 * s1, z_j2],
            [r_j3 * c1, r_j3 * s1, z_j3],
            [r_ee * c1, r_ee * s1, z_ee],
        ])

    def jacobians(self, q):
        """返回 (3, 3, 3) 数组：J[i] 是连杆 i COM 的 3×3 Jacobian

        J[i]_{jk} = ∂(COM_i)_j / ∂q_k
        """
        th1, th2, th3 = q
        c1, s1 = np.cos(th1), np.sin(th1)
        c2, s2 = np.cos(th2), np.sin(th2)
        c23, s23 = np.cos(th2 + th3), np.sin(th2 + th3)
        a1, a2 = self.a1, self.a2
        l1, l2 = self.l1, self.l2
        leff = l2 + self.a3

        J = np.zeros((3, 3, 3))

        # Link 1 COM: r1 = a1*c2
        J[0, 0, 0] = -a1 * c2 * s1
        J[0, 1, 0] = a1 * c2 * c1
        J[0, 2, 0] = 0.0
        J[0, 0, 1] = -a1 * s2 * c1
        J[0, 1, 1] = -a1 * s2 * s1
        J[0, 2, 1] = a1 * c2

        # Link 2 COM: r2 = l1*c2 + a2*c23
        r2 = l1 * c2 + a2 * c23
        J[1, 0, 0] = -r2 * s1
        J[1, 1, 0] = r2 * c1
        J[1, 2, 0] = 0.0
        J[1, 0, 1] = (-l1 * s2 - a2 * s23) * c1
        J[1, 1, 1] = (-l1 * s2 - a2 * s23) * s1
        J[1, 2, 1] = l1 * c2 + a2 * c23
        J[1, 0, 2] = (-a2 * s23) * c1
        J[1, 1, 2] = (-a2 * s23) * s1
        J[1, 2, 2] = a2 * c23

        # Link 3 COM: r3 = l1*c2 + leff*c23
        r3 = l1 * c2 + leff * c23
        J[2, 0, 0] = -r3 * s1
        J[2, 1, 0] = r3 * c1
        J[2, 2, 0] = 0.0
        J[2, 0, 1] = (-l1 * s2 - leff * s23) * c1
        J[2, 1, 1] = (-l1 * s2 - leff * s23) * s1
        J[2, 2, 1] = l1 * c2 + leff * c23
        J[2, 0, 2] = (-leff * s23) * c1
        J[2, 1, 2] = (-leff * s23) * s1
        J[2, 2, 2] = leff * c23

        return J


class ArmDynamics:
    """3-DOF 臂动力学：M(q) ddq + C(q,dq) dq + G(q) = τ

    点质量近似：M = Σ mᵢ Jᵥᵢᵀ Jᵥᵢ
    C 由 Christoffel 符号通过有限差分计算。
    """

    def __init__(self, m1=2.0, m2=1.5, m3=1.0, l1=0.4, l2=0.3, l3=0.2, g=9.81):
        self.m1, self.m2, self.m3 = m1, m2, m3
        self.g = g
        self.masses = np.array([m1, m2, m3])
        self.fk = ForwardKinematics(l1, l2, l3)

    def mass_matrix(self, q):
        """M(q) = Σ mᵢ Jᵥᵢᵀ Jᵥᵢ"""
        J = self.fk.jacobians(q)
        M = np.zeros((3, 3))
        for i in range(3):
            Ji = J[i]
            M += self.masses[i] * Ji.T @ Ji
        return M

    def gravity_vector(self, q):
        """Gₖ = ∂V/∂qₖ = Σ mᵢ g (∂zᵢ/∂qₖ)"""
        J = self.fk.jacobians(q)
        G = np.zeros(3)
        for i in range(3):
            G += self.masses[i] * self.g * J[i, 2, :]
        return G

    def _grad_M(self, q):
        """返回 (3, 3, 3) 张量: dM[i,j,k] = ∂M_{ij}/∂q_k (有限差分)"""
        eps = 1e-6
        M0 = self.mass_matrix(q)
        dM = np.zeros((3, 3, 3))
        for k in range(3):
            q_plus = q.copy()
            q_plus[k] += eps
            M_plus = self.mass_matrix(q_plus)
            dM[:, :, k] = (M_plus - M0) / eps
        return dM

    def coriolis_torque(self, q, dq):
        """c = C(q,dq) @ dq

        c_k = Σ_{i,j} Γ_{ijk} dq_i dq_j
        Γ_{ijk} = 0.5 * (∂M_{kj}/∂q_i + ∂M_{ki}/∂q_j - ∂M_{ij}/∂q_k)
        """
        dM = self._grad_M(q)
        c = np.zeros(3)
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    gamma = 0.5 * (dM[k, j, i] + dM[k, i, j] - dM[i, j, k])
                    c[k] += gamma * dq[i] * dq[j]
        return c

    def forward_dynamics(self, q, dq, tau):
        """ddq = M⁻¹ (τ - C(q,dq) dq - G(q))"""
        M = self.mass_matrix(q)
        c = self.coriolis_torque(q, dq)
        G = self.gravity_vector(q)
        ddq = np.linalg.solve(M, tau - c - G)
        return ddq

    def step(self, q, dq, tau, dt):
        """RK4 积分一步，返回 (q_new, dq_new)"""
        def rhs(state, tau_val):
            qq = state[:3]
            dqq = state[3:]
            ddqq = self.forward_dynamics(qq, dqq, tau_val)
            return np.concatenate([dqq, ddqq])

        state = np.concatenate([q, dq])
        k1 = rhs(state, tau)
        k2 = rhs(state + 0.5 * dt * k1, tau)
        k3 = rhs(state + 0.5 * dt * k2, tau)
        k4 = rhs(state + dt * k3, tau)
        new_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_state[:3], new_state[3:]

    def inverse_dynamics(self, q, dq, ddq):
        """τ = M(q) ddq + C(q,dq) dq + G(q)"""
        M = self.mass_matrix(q)
        c = self.coriolis_torque(q, dq)
        G = self.gravity_vector(q)
        return M @ ddq + c + G
