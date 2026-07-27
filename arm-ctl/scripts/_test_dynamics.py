"""验证 forward_dynamics 和 inverse_dynamics 互为逆运算"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from arm import ArmDynamics

dyn = ArmDynamics()
q = np.array([0.5, 0.3, -0.8])
dq = np.array([0.1, -0.2, 0.15])
ddq_true = np.array([1.0, -2.0, 3.0])

tau = dyn.inverse_dynamics(q, dq, ddq_true)
ddq_recovered = dyn.forward_dynamics(q, dq, tau)

print(f"M(q) 条件数: {np.linalg.cond(dyn.mass_matrix(q)):.2f}")
print(f"G(q): {dyn.gravity_vector(q)}")
print(f"C(q,dq)*dq: {dyn.coriolis_torque(q, dq)}")
print(f"τ = {tau}")
print(f"ddq_true = {ddq_true}")
print(f"ddq_recovered = {ddq_recovered}")
print(f"误差: {np.max(np.abs(ddq_recovered - ddq_true)):.2e}")
assert np.allclose(ddq_recovered, ddq_true, atol=1e-4), "动力学一致性检查失败!"
print("PASS: 动力学一致性验证通过")
