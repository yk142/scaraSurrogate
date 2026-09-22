import numpy as np

from src.control import angular_error
from src.control_3d import run_ptp_true


def test_ptp_controller_reaches_target_on_true_plant():
    """3D空間マニピュレータの真の物理モデルに対し、計算トルク法PTP制御が目標角度に収束すること。"""
    target = np.array([0.5, 0.5, -0.5])
    initial_state = np.array([-0.5, -0.3, 0.1, 0.0, 0.0, 0.0])

    traj = run_ptp_true(initial_state, target, n_steps=1500, dt=0.02)  # 30秒

    final_error = np.abs(angular_error(traj[-50:, :3], target)).sum(axis=-1)
    assert final_error.mean() < 0.1


def test_ptp_controller_reaches_target_from_large_offset():
    """大オフセットICでも収束すること。"""
    target = np.array([0.5, 0.5, -0.5])
    initial_state = np.array([1.5, 1.3, -1.1, 0.0, 0.0, 0.0])

    traj = run_ptp_true(initial_state, target, n_steps=1500, dt=0.02)

    final_error = np.abs(angular_error(traj[-50:, :3], target)).sum(axis=-1)
    assert final_error.mean() < 0.1
