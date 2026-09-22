import numpy as np
import torch

from src.model_3d import (
    NSSModel3D,
    StructuredFrictionGrayBoxModel3D,
    coriolis_matrix_3d_torch,
    gravity_vector_3d_torch,
    mass_matrix_3d_torch,
)
from src.physics_3d import coriolis_matrix_3d, gravity_vector_3d, mass_matrix_3d, rk4_step


def test_mass_matrix_matches_numpy():
    q1, q2 = 0.4, -1.2
    M_np = mass_matrix_3d(np.array(q1), np.array(q2))
    M_torch = mass_matrix_3d_torch(torch.tensor(q1), torch.tensor(q2)).numpy()
    assert np.allclose(M_np, M_torch, atol=1e-6)


def test_coriolis_matrix_matches_numpy():
    q1, q2 = 0.4, -1.2
    q0_dot, q1_dot, q2_dot = 0.3, 0.5, -0.8
    C_np = coriolis_matrix_3d(np.array(q1), np.array(q2), np.array(q0_dot), np.array(q1_dot), np.array(q2_dot))
    C_torch = coriolis_matrix_3d_torch(
        torch.tensor(q1), torch.tensor(q2), torch.tensor(q0_dot), torch.tensor(q1_dot), torch.tensor(q2_dot)
    ).numpy()
    assert np.allclose(C_np, C_torch, atol=1e-6)


def test_gravity_vector_matches_numpy():
    q1, q2 = 0.4, -1.2
    g_np = gravity_vector_3d(np.array(q1), np.array(q2))
    g_torch = gravity_vector_3d_torch(torch.tensor(q1), torch.tensor(q2)).numpy()
    assert np.allclose(g_np, g_torch, atol=1e-6)


def _zero_residual_graybox() -> StructuredFrictionGrayBoxModel3D:
    model = StructuredFrictionGrayBoxModel3D()
    with torch.no_grad():
        model.log_c_viscous.fill_(-30.0)  # exp(-30) ~= 0
        model.log_c_coulomb.fill_(-30.0)
    return model


def test_zero_residual_matches_known_physics_without_friction():
    """残差がほぼ0なら、グレーボックスの1ステップは無摩擦の既知物理RK4と一致すること。"""
    model = _zero_residual_graybox()
    dt = model.dt
    zero_friction = np.zeros(3)

    for state_np, tau_np in [
        ([0.1, 0.3, -0.5, 0.2, 1.0, -0.7], [0.0, 0.0, 0.0]),
        ([-1.0, -np.pi / 2 + 0.2, 0.3, 0.5, -1.5, 0.9], [0.3, 0.5, -0.3]),
    ]:
        state = torch.tensor(state_np, dtype=torch.float32)
        u = torch.tensor(tau_np, dtype=torch.float32)

        with torch.no_grad():
            predicted = model.step(state, u).numpy()
        expected = rk4_step(
            np.array(state_np), dt, np.array(tau_np), c_viscous=zero_friction, c_coulomb=zero_friction
        )

        assert np.allclose(predicted, expected, atol=1e-3)


def test_graybox_3d_rollout_output_shape():
    model = StructuredFrictionGrayBoxModel3D()
    traj = model.rollout(np.array([0.1, 0.2, -0.3, 0.0, 0.0, 0.0]), n_steps=10)
    assert traj.shape == (11, 6)


def test_nss_3d_rollout_output_shape():
    model = NSSModel3D()
    traj = model.rollout(np.array([0.1, 0.2, -0.3, 0.0, 0.0, 0.0]), n_steps=10)
    assert traj.shape == (11, 6)
