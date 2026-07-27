"""验证 PID 和 CTC 控制器的基本行为"""
import numpy as np
from arm import ArmDynamics
from controllers import PIDController, CTCController

dyn = ArmDynamics()
q = np.array([0.0, 0.0, 0.0])
dq = np.zeros(3)
q_des = np.array([0.5, 0.3, -0.8])
dq_des = np.zeros(3)
ddq_des = np.zeros(3)

pid = PIDController()
tau_pid = pid.compute_torque(0, q, dq, q_des, dq_des, ddq_des)
print(f"PID torque with error: {tau_pid}")
assert np.any(np.abs(tau_pid) > 0), "PID should produce non-zero torque with error!"

pid.reset()
tau_pid_zero = pid.compute_torque(0, q_des, dq, q_des, dq_des, ddq_des)
print(f"PID torque at zero error (fresh reset): {tau_pid_zero}")
assert np.allclose(tau_pid_zero, 0, atol=1e-10)

ctc = CTCController(dyn)
tau_ctc = ctc.compute_torque(0, q, dq, q, dq, ddq_des)
print(f"CTC gravity compensation at q=0: {tau_ctc}")
G = dyn.gravity_vector(q)
print(f"Expected gravity at q=0: {G}")
assert np.allclose(tau_ctc, G, atol=0.1)

tau_ctc_err = ctc.compute_torque(0, q, dq, q_des, dq_des, ddq_des)
print(f"CTC torque with error: {tau_ctc_err}")
assert np.any(np.abs(tau_ctc_err) > np.abs(G))
print("PASS: 控制器基本行为验证通过")
