import numpy as np

from src.evaluate_transient import settling_time, trajectory_error


def test_settling_time_finds_last_threshold_crossing():
    target = np.array([0.0, 0.0])
    dt = 0.1
    # 誤差(joint1+joint2の絶対値和): 1.0, 1.0, 0.05, 0.2, 0.05, 0.05 -> しきい値0.1を
    # 最後に上回るのはindex 3、以降ずっと下回るので整定時刻はindex 4 * dt = 0.4
    q1 = np.array([1.0, 1.0, 0.05, 0.2, 0.05, 0.05])
    traj = np.stack([q1, np.zeros_like(q1), np.zeros_like(q1), np.zeros_like(q1)], axis=-1)

    t = settling_time(traj, target, dt, threshold=0.1)
    assert np.isclose(t, 0.4)


def test_settling_time_nan_when_never_settles():
    target = np.array([0.0, 0.0])
    dt = 0.1
    q1 = np.array([1.0, 0.5, 0.3])
    traj = np.stack([q1, np.zeros_like(q1), np.zeros_like(q1), np.zeros_like(q1)], axis=-1)

    t = settling_time(traj, target, dt, threshold=0.1)
    assert np.isnan(t)


def test_trajectory_error_is_zero_for_identical_trajectories():
    traj = np.array([[0.5, -0.3, 0.0, 0.0], [0.4, -0.2, 0.1, -0.1]])
    err = trajectory_error(traj, traj)
    assert np.allclose(err, 0.0)
