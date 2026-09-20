import numpy as np

from src.physics import kinetic_energy, simulate


def test_energy_conservation_no_torque_no_friction():
    """無トルク・無摩擦では、水平面のため全運動エネルギーが保存されること。"""
    initial_state = np.array([0.3, -0.5, 1.0, -0.7])
    dt = 0.005
    n_steps = 4000  # 20秒分

    traj = simulate(initial_state, dt, n_steps, tau=None, c=0.0)

    e = kinetic_energy(traj)
    e0 = e[0]
    max_rel_drift = np.max(np.abs(e - e0)) / e0

    assert max_rel_drift < 1e-3


def test_rest_state_is_fixed_point():
    """q_dot=0, tau=0 は不動点であること(水平面なので重力による復元力がない)。"""
    initial_state = np.array([0.4, -0.9, 0.0, 0.0])
    traj = simulate(initial_state, dt=0.01, n_steps=200, tau=None, c=0.0)
    assert np.allclose(traj, initial_state)


def test_friction_dissipates_energy():
    """粘性摩擦(c>0)があれば運動エネルギーが単調に(非増加に)減少すること。"""
    initial_state = np.array([0.0, 0.5, 2.0, -1.5])
    dt = 0.005
    n_steps = 2000

    traj = simulate(initial_state, dt, n_steps, tau=None, c=0.5)

    e = kinetic_energy(traj)
    assert np.all(np.diff(e) <= 1e-9)
    assert e[-1] < e[0]


def test_constant_torque_increases_energy_when_aligned_with_motion():
    """静止状態に一定トルクを与えれば、運動エネルギーが増加すること(仕事をする)。"""
    initial_state = np.array([0.0, 0.3, 0.0, 0.0])
    dt = 0.005
    n_steps = 200
    tau = np.array([1.0, 0.5])

    traj = simulate(initial_state, dt, n_steps, tau=tau, c=0.0)

    e = kinetic_energy(traj)
    assert e[-1] > e[0]
