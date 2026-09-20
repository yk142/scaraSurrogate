"""自己回帰ロールアウトと誤差指標の計算。"""
import numpy as np

from src.physics import simulate


def true_rollout(
    initial_states: np.ndarray, dt: float, n_steps: int, tau_seq: np.ndarray | None = None
) -> np.ndarray:
    """shape (n_ic, n_steps+1, 4)

    tau_seq: shape (n_steps, 2)。全ICに共通のトルク列として適用する。
    """
    return np.stack([simulate(s0, dt, n_steps, tau=tau_seq) for s0 in initial_states], axis=0)


def rmse_curve(true_traj: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
    """horizon step ごとのRMSEを返す(角度2次元はwrap済み差、角速度2次元は単純差)。

    true_traj, pred_traj: shape (n_steps+1, n_ic, 4)

    Returns: shape (n_steps+1,)
    """
    angle_err = np.arctan2(
        np.sin(true_traj[..., :2] - pred_traj[..., :2]),
        np.cos(true_traj[..., :2] - pred_traj[..., :2]),
    )
    vel_err = true_traj[..., 2:] - pred_traj[..., 2:]
    sq_err = (angle_err**2).sum(axis=-1) + (vel_err**2).sum(axis=-1)  # (n_steps+1, n_ic)
    return np.sqrt(sq_err.mean(axis=1))
