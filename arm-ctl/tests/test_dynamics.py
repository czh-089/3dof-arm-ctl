"""测试 ArmDynamics 正逆动力学一致性"""
import numpy as np


def test_mass_matrix_positive_definite(dyn, sample_state):
    q, _, _ = sample_state
    M = dyn.mass_matrix(q)
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0), f"Mass matrix not positive definite: {eigvals}"


def test_gravity_at_rest(dyn):
    q = np.zeros(3)
    G = dyn.gravity_vector(q)
    # 在 q=0 时，只有关节 2 和 3 承受重力
    assert abs(G[0]) < 1e-10, f"θ₁ should have zero gravity torque, got {G[0]}"
    assert G[1] > 0, f"θ₂ should have positive gravity torque, got {G[1]}"
    assert G[2] < G[1], f"θ₃ gravity should be less than θ₂, got {G[2]}"


def test_forward_inverse_consistency(dyn, sample_state):
    q, dq, ddq_true = sample_state
    tau = dyn.inverse_dynamics(q, dq, ddq_true)
    ddq_recovered = dyn.forward_dynamics(q, dq, tau)
    assert np.allclose(ddq_recovered, ddq_true, atol=1e-4), \
        f"Roundtrip failed: max error {np.max(np.abs(ddq_recovered - ddq_true)):.2e}"


def test_coriolis_zero_at_zero_velocity(dyn):
    q = np.array([0.5, 0.3, -0.8])
    dq = np.zeros(3)
    c = dyn.coriolis_torque(q, dq)
    assert np.allclose(c, 0, atol=1e-10), \
        f"Coriolis should be zero at zero velocity, got {c}"


def test_rk4_energy_conservation(dyn):
    q0 = np.array([0.3, 0.6, -1.0])
    dq0 = np.zeros(3)
    q, dq = q0.copy(), dq0.copy()
    dt = 0.001

    M0 = dyn.mass_matrix(q0)
    E0 = 0.5 * dq0 @ M0 @ dq0
    coms = dyn.fk.com_positions(q0)
    E0 += dyn.g * np.sum(dyn.masses * coms[:, 2])

    for _ in range(1000):
        q, dq = dyn.step(q, dq, np.zeros(3), dt)

    M = dyn.mass_matrix(q)
    E = 0.5 * dq @ M @ dq
    coms = dyn.fk.com_positions(q)
    E += dyn.g * np.sum(dyn.masses * coms[:, 2])

    drift = abs(E - E0) / abs(E0)
    assert drift < 0.01, f"Energy drift {drift*100:.3f}% too large"
