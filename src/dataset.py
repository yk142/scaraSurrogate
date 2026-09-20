"""学習・評価用の軌道データセット生成。

姉妹プロジェクト surrogateRolloutVerification のM7で「マルチステップ(ロールアウト)
損失にはKステップの学習ウィンドウが必要」と分かっているため、最初からウィンドウ
生成を組み込む(1-step用データセットは作らない)。
"""
import numpy as np

from src.physics import simulate

THETA_RANGE = (-np.pi, np.pi)
THETA_DOT_RANGE = (-3.0, 3.0)

TAU_RANGE = (-1.0, 1.0)
TORQUE_HOLD_STEPS = 5  # このステップ数ごとにトルクを変更する(区分定数=ゼロ次ホールド)


def sample_initial_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """IC を一様サンプリングする。shape (n, 4) = [theta1, theta2, theta1_dot, theta2_dot]"""
    theta = rng.uniform(*THETA_RANGE, size=(n, 2))
    theta_dot = rng.uniform(*THETA_DOT_RANGE, size=(n, 2))
    return np.concatenate([theta, theta_dot], axis=-1)


def sample_torque_sequence(
    n_steps: int,
    rng: np.random.Generator,
    tau_range: tuple[float, float] = TAU_RANGE,
    hold_steps: int = TORQUE_HOLD_STEPS,
) -> np.ndarray:
    """2軸分のランダムな区分定数トルク列を生成する。shape (n_steps, 2)"""
    n_holds = int(np.ceil(n_steps / hold_steps))
    hold_values = rng.uniform(*tau_range, size=(n_holds, 2))
    return np.repeat(hold_values, hold_steps, axis=0)[:n_steps]


def generate_controlled_trajectories(
    n_trajectories: int, dt: float, n_steps: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """ランダムトルク列で駆動した軌道データセットを生成する。

    Returns:
        trajectories: shape (n_traj, n_steps + 1, 4)
        tau_seqs:     shape (n_traj, n_steps, 2)
    """
    rng = np.random.default_rng(seed)
    initial_states = sample_initial_states(n_trajectories, rng)
    tau_seqs = np.stack(
        [sample_torque_sequence(n_steps, rng) for _ in range(n_trajectories)], axis=0
    )
    trajectories = np.stack(
        [simulate(s0, dt, n_steps, tau=tau_seq) for s0, tau_seq in zip(initial_states, tau_seqs)],
        axis=0,
    )
    return trajectories, tau_seqs


def make_rollout_windows(
    trajectories: np.ndarray, tau_seqs: np.ndarray, k: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """軌道群+トルク列からKステップの学習ウィンドウ (x0, u_seq, targets) を作る。

    trajectories: shape (n_traj, n_steps + 1, 4), tau_seqs: shape (n_traj, n_steps, 2)

    Returns:
        x0:      shape (n_windows, 4)       各ウィンドウの初期状態
        u_seq:   shape (k, n_windows, 2)    各ウィンドウのKステップ分トルク列
        targets: shape (k, n_windows, 4)    各ウィンドウのKステップ分正解状態
    """
    n_traj, n_steps_plus_1, _ = trajectories.shape
    n_steps = n_steps_plus_1 - 1

    x0_parts, u_parts, target_parts = [], [], []
    for start in range(0, n_steps - k + 1, stride):
        x0_parts.append(trajectories[:, start, :])
        u_parts.append(tau_seqs[:, start : start + k, :])
        target_parts.append(trajectories[:, start + 1 : start + k + 1, :])

    x0 = np.concatenate(x0_parts, axis=0)
    u_seq = np.concatenate(u_parts, axis=0).transpose(1, 0, 2)
    targets = np.concatenate(target_parts, axis=0).transpose(1, 0, 2)
    return x0, u_seq, targets
