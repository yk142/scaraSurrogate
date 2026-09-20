import numpy as np
import torch

from src.model import GrayBoxModel, NSSModel, coriolis_matrix_torch, mass_matrix_torch
from src.physics import coriolis_matrix, mass_matrix, rk4_step


def test_mass_and_coriolis_matrices_match_numpy():
    """torch版のM(q), C(q,q_dot)がnumpy版(physics.py)と数値的に一致すること。"""
    q2 = 1.234
    q1_dot, q2_dot = 0.5, -0.8

    M_np = mass_matrix(np.array(q2))
    C_np = coriolis_matrix(np.array(q2), np.array(q1_dot), np.array(q2_dot))

    M_torch = mass_matrix_torch(torch.tensor(q2)).numpy()
    C_torch = coriolis_matrix_torch(
        torch.tensor(q2), torch.tensor(q1_dot), torch.tensor(q2_dot)
    ).numpy()

    assert np.allclose(M_np, M_torch, atol=1e-6)
    assert np.allclose(C_np, C_torch, atol=1e-6)


def _zero_residual_graybox() -> GrayBoxModel:
    model = GrayBoxModel()
    final_layer = model.net[-1]
    torch.nn.init.zeros_(final_layer.weight)
    torch.nn.init.zeros_(final_layer.bias)
    return model


def test_zero_residual_matches_known_physics_without_friction():
    """残差が0なら、グレーボックスの1ステップは無摩擦の既知物理RK4と一致すること。"""
    model = _zero_residual_graybox()
    dt = model.dt
    zero_friction = np.zeros(2)

    for state_np, tau_np in [
        ([0.3, -0.5, 1.0, -0.7], [0.0, 0.0]),
        ([np.pi - 0.2, 2.0, -1.5, 0.9], [0.5, -0.3]),
    ]:
        state = torch.tensor(state_np, dtype=torch.float32)
        u = torch.tensor(tau_np, dtype=torch.float32)

        with torch.no_grad():
            predicted = model.step(state, u).numpy()
        expected = rk4_step(
            np.array(state_np), dt, np.array(tau_np), c_viscous=zero_friction, c_coulomb=zero_friction
        )

        assert np.allclose(predicted, expected, atol=1e-4)


def test_nss_rollout_output_shape():
    model = NSSModel()
    traj = model.rollout(np.array([0.1, 0.2, 0.0, 0.0]), n_steps=10)
    assert traj.shape == (11, 4)


def test_graybox_rollout_output_shape():
    model = GrayBoxModel()
    traj = model.rollout(np.array([0.1, 0.2, 0.0, 0.0]), n_steps=10)
    assert traj.shape == (11, 4)
