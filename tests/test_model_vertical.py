import numpy as np
import torch

from src.model_vertical import GrayBoxModelVertical, StructuredFrictionGrayBoxModelVertical, gravity_vector_torch
from src.physics_vertical import friction_torque, gravity_vector, rk4_step


def test_gravity_vector_matches_numpy():
    """torch版G(q)がnumpy版(physics_vertical.py)と数値的に一致すること。"""
    q1, q2 = 0.4, -1.2

    g_np = gravity_vector(np.array(q1), np.array(q2))
    g_torch = gravity_vector_torch(torch.tensor(q1), torch.tensor(q2)).numpy()

    assert np.allclose(g_np, g_torch, atol=1e-6)


def _zero_residual_graybox() -> GrayBoxModelVertical:
    model = GrayBoxModelVertical()
    final_layer = model.net[-1]
    torch.nn.init.zeros_(final_layer.weight)
    torch.nn.init.zeros_(final_layer.bias)
    return model


def test_zero_residual_matches_known_physics_without_friction():
    """残差が0なら、グレーボックスの1ステップは無摩擦の既知物理(重力あり)RK4と一致すること。"""
    model = _zero_residual_graybox()
    dt = model.dt
    zero_friction = np.zeros(2)

    for state_np, tau_np in [
        ([0.3, -0.5, 1.0, -0.7], [0.0, 0.0]),
        ([-np.pi / 2, 0.3, -1.5, 0.9], [0.5, -0.3]),
    ]:
        state = torch.tensor(state_np, dtype=torch.float32)
        u = torch.tensor(tau_np, dtype=torch.float32)

        with torch.no_grad():
            predicted = model.step(state, u).numpy()
        expected = rk4_step(
            np.array(state_np), dt, np.array(tau_np), c_viscous=zero_friction, c_coulomb=zero_friction
        )

        assert np.allclose(predicted, expected, atol=1e-4)


def test_graybox_vertical_rollout_output_shape():
    model = GrayBoxModelVertical()
    traj = model.rollout(np.array([0.1, 0.2, 0.0, 0.0]), n_steps=10)
    assert traj.shape == (11, 4)


def test_structured_friction_matches_true_friction_when_coefficients_match():
    """Phase2-M8 (#37): 係数がtrue値と一致すれば、構造化残差は真の摩擦トルクと一致すること。"""
    model = StructuredFrictionGrayBoxModelVertical()
    with torch.no_grad():
        model.log_c_viscous.copy_(torch.log(torch.tensor([1.0, 1.0])))
        model.log_c_coulomb.copy_(torch.log(torch.tensor([0.02, 0.02])))

    q_dot_np = np.array([0.5, -1.2])
    state = torch.tensor([0.3, -0.5, *q_dot_np], dtype=torch.float32)
    u = torch.zeros(2)

    with torch.no_grad():
        predicted = model.residual_torque(state, u).numpy()
    expected = friction_torque(q_dot_np, c_viscous=np.array([1.0, 1.0]), c_coulomb=np.array([0.02, 0.02]))

    assert np.allclose(predicted, expected, atol=1e-5)


def test_structured_friction_rollout_output_shape():
    model = StructuredFrictionGrayBoxModelVertical()
    traj = model.rollout(np.array([0.1, 0.2, 0.0, 0.0]), n_steps=10)
    assert traj.shape == (11, 4)
