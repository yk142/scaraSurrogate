import numpy as np

from src.control import angular_error, run_ptp_true


def test_ptp_controller_reaches_target_on_true_plant():
    """真の物理モデルに対し、計算トルク法PTP制御が目標角度に収束すること。"""
    target = np.array([0.5, -0.5])
    initial_state = np.array([-0.5, 0.5, 0.0, 0.0])

    traj = run_ptp_true(initial_state, target, n_steps=1500, dt=0.02)  # 30秒

    final_error = np.abs(angular_error(traj[-50:, :2], target)).sum(axis=-1)
    assert final_error.mean() < 0.1


def test_ptp_controller_does_not_stick_from_friction():
    """M3 (#5) で摩擦固着によって目標に到達できなかったICが、INTEGRAL_CLIP引き上げ後
    (M4 #7) は到達できること。"""
    target = np.array([0.5, -0.5])
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # M3で固着した代表IC(offset=(-0.5,+0.5))

    traj = run_ptp_true(initial_state, target, n_steps=1500, dt=0.02)  # 30秒

    final_error = np.abs(angular_error(traj[-50:, :2], target)).sum(axis=-1)
    assert final_error.mean() < 0.1
