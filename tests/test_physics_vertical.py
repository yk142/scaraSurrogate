import numpy as np

from src.physics_vertical import gravity_vector, simulate, total_energy

ZERO_FRICTION = np.zeros(2)


def test_energy_conservation_no_torque_no_friction():
    """無トルク・無摩擦では、重力ありでも力学的全エネルギー(運動+位置)が保存されること。"""
    initial_state = np.array([0.3, -0.5, 1.0, -0.7])
    dt = 0.005
    n_steps = 4000  # 20秒分

    traj = simulate(initial_state, dt, n_steps, tau=None, c_viscous=ZERO_FRICTION, c_coulomb=ZERO_FRICTION)

    e = total_energy(traj)
    e0 = e[0]
    max_rel_drift = np.max(np.abs(e - e0)) / np.abs(e0)

    assert max_rel_drift < 1e-3


def test_hanging_down_is_fixed_point():
    """鉛直下向き(q1=-pi/2, q2=0)は重力トルクが0になる不動点であること。"""
    initial_state = np.array([-np.pi / 2, 0.0, 0.0, 0.0])
    traj = simulate(initial_state, dt=0.01, n_steps=200, tau=None, c_viscous=ZERO_FRICTION, c_coulomb=ZERO_FRICTION)
    assert np.allclose(traj, initial_state, atol=1e-9)


def test_gravity_vector_zero_at_hanging_equilibrium():
    g = gravity_vector(np.array(-np.pi / 2), np.array(0.0))
    assert np.allclose(g, 0.0, atol=1e-9)


def test_friction_dissipates_mechanical_energy():
    """粘性+クーロン摩擦があれば力学的全エネルギーが単調に(非増加に)減少すること。"""
    initial_state = np.array([0.0, 0.5, 2.0, -1.5])
    dt = 0.005
    n_steps = 2000

    traj = simulate(initial_state, dt, n_steps, tau=None)  # デフォルトの摩擦係数を使用

    e = total_energy(traj)
    assert np.all(np.diff(e) <= 1e-6)
