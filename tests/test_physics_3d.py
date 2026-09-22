import numpy as np

from src.physics_3d import mass_matrix_3d, simulate, total_energy

ZERO_FRICTION = np.zeros(3)


def test_energy_conservation_no_torque_no_friction():
    """無トルク・無摩擦では、ヨー運動を含めても力学的全エネルギーが保存されること。"""
    initial_state = np.array([0.2, 0.3, -0.5, 0.8, 1.0, -0.7])
    dt = 0.005
    n_steps = 4000  # 20秒分

    traj = simulate(initial_state, dt, n_steps, tau=None, c_viscous=ZERO_FRICTION, c_coulomb=ZERO_FRICTION)

    e = total_energy(traj)
    e0 = e[0]
    max_rel_drift = np.max(np.abs(e - e0)) / np.abs(e0)

    assert max_rel_drift < 1e-3


def test_hanging_down_with_no_yaw_velocity_is_fixed_point():
    """鉛直下向き(q1=-pi/2, q2=0)かつヨー速度0なら不動点であること。"""
    initial_state = np.array([0.5, -np.pi / 2, 0.0, 0.0, 0.0, 0.0])
    traj = simulate(initial_state, dt=0.01, n_steps=200, tau=None, c_viscous=ZERO_FRICTION, c_coulomb=ZERO_FRICTION)
    assert np.allclose(traj, initial_state, atol=1e-9)


def test_energy_conserved_through_kinematic_singularity():
    """腕がヨー軸と一直線になる運動学的特異点(M00→0)を通過する軌道でも、
    M00_EPSILON正則化により力学的全エネルギーが保存されること(Phase3-M2 #52)。
    """
    # q1=-pi/2, q2=0 は腕がヨー軸と一直線(M00がM00_EPSILON近くまで下がる)。
    initial_state = np.array([0.0, -np.pi / 2 + 0.05, 0.0, 0.5, 0.0, 0.0])
    dt = 0.005
    n_steps = 2000  # 10秒分

    traj = simulate(initial_state, dt, n_steps, tau=None, c_viscous=ZERO_FRICTION, c_coulomb=ZERO_FRICTION)

    m00 = mass_matrix_3d(traj[:, 1], traj[:, 2])[:, 0, 0]
    assert m00.min() < 0.05  # 特異点付近を実際に通過していることを確認

    e = total_energy(traj)
    max_rel_drift = np.max(np.abs(e - e[0])) / np.abs(e[0])
    assert max_rel_drift < 1e-3


def test_mass_matrix_is_block_diagonal():
    """質量行列はヨー(q0)とピッチ(q1,q2)についてブロック対角であること。"""
    M = mass_matrix_3d(np.array(0.4), np.array(-0.6))
    assert np.allclose(M[0, 1], 0.0)
    assert np.allclose(M[0, 2], 0.0)
    assert np.allclose(M[1, 0], 0.0)
    assert np.allclose(M[2, 0], 0.0)


def test_reduces_to_planar_dynamics_when_yaw_velocity_is_zero():
    """ヨー速度0なら、q1,q2の運動はPhase2の平面2リンクアームと一致すること。"""
    from src.physics_vertical import simulate as simulate_vertical

    initial_state_3d = np.array([0.3, 0.2, -0.5, 0.0, 1.0, -0.7])
    initial_state_2d = np.array([0.2, -0.5, 1.0, -0.7])
    dt = 0.01
    n_steps = 200

    traj_3d = simulate(initial_state_3d, dt, n_steps, tau=None, c_viscous=ZERO_FRICTION, c_coulomb=ZERO_FRICTION)
    traj_2d = simulate_vertical(
        initial_state_2d, dt, n_steps, tau=None, c_viscous=np.zeros(2), c_coulomb=np.zeros(2)
    )

    assert np.allclose(traj_3d[:, 0], 0.3, atol=1e-9)  # ヨー角は変化しない
    assert np.allclose(traj_3d[:, 1:3], traj_2d[:, 0:2], atol=1e-6)
    assert np.allclose(traj_3d[:, 4:6], traj_2d[:, 2:4], atol=1e-6)
