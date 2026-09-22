"""Phase3: ヨー+肩・肘の3D空間マニピュレータ用の学習・評価用データセット生成。

`make_rollout_windows`(`src/dataset.py`)は次元に依存しないためそのまま
再利用できるが、IC・トルクのサンプリングは3軸分に拡張する必要がある。

TAU_RANGE, VELOCITY_LIMIT: Phase2-M2/M5/M13の教訓を踏まえ、重力補償に必要な
トルク・カオス対策の角速度上限をPhase2と同じ水準にする。
"""
import numpy as np

from src.physics_3d import simulate

THETA_RANGE = (-np.pi, np.pi)
THETA_DOT_RANGE = (-3.0, 3.0)

TAU_RANGE = (-10.0, 10.0)
TORQUE_HOLD_STEPS = 5

# Phase3-M2 (#52): 腕がヨー軸と一直線に伸びる運動学的特異点付近ではM00(ヨー軸
# まわりの慣性)が0に近づくが、physics_3d.M00_FLOORによる正則化で発散は防止
# 済み(実機の特異点処理と同種)。角速度上限による棄却サンプリングはPhase2-M2
# と同じくカオス対策として引き続き用いる。
VELOCITY_LIMIT = 10.0
MAX_RESAMPLE_ATTEMPTS = 20


def sample_initial_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """ICを一様サンプリングする。shape (n, 6) = [q0,q1,q2,q0_dot,q1_dot,q2_dot]"""
    theta = rng.uniform(*THETA_RANGE, size=(n, 3))
    theta_dot = rng.uniform(*THETA_DOT_RANGE, size=(n, 3))
    return np.concatenate([theta, theta_dot], axis=-1)


def sample_torque_sequence(
    n_steps: int,
    rng: np.random.Generator,
    tau_range: tuple[float, float] = TAU_RANGE,
    hold_steps: int = TORQUE_HOLD_STEPS,
) -> np.ndarray:
    """3軸分のランダムな区分定数トルク列を生成する。shape (n_steps, 3)"""
    n_holds = int(np.ceil(n_steps / hold_steps))
    hold_values = rng.uniform(*tau_range, size=(n_holds, 3))
    return np.repeat(hold_values, hold_steps, axis=0)[:n_steps]


def _sample_one_trajectory(dt: float, n_steps: int, tau_range: tuple[float, float], rng: np.random.Generator):
    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        ic = sample_initial_states(1, rng)[0]
        tau_seq = sample_torque_sequence(n_steps, rng, tau_range=tau_range)
        traj = simulate(ic, dt, n_steps, tau=tau_seq)
        if np.all(np.isfinite(traj)) and np.abs(traj[:, 3:]).max() <= VELOCITY_LIMIT:
            return traj, tau_seq
    return traj, tau_seq


def generate_controlled_trajectories(
    n_trajectories: int, dt: float, n_steps: int, seed: int, tau_range: tuple[float, float] = TAU_RANGE
) -> tuple[np.ndarray, np.ndarray]:
    """ランダムトルク列で駆動した軌道データセットを生成する。

    角速度が`VELOCITY_LIMIT`を超える軌道は棄却して引き直す(Phase2-M2参照)。

    Returns:
        trajectories: shape (n_traj, n_steps + 1, 6)
        tau_seqs:     shape (n_traj, n_steps, 3)
    """
    rng = np.random.default_rng(seed)
    trajectories, tau_seqs = zip(
        *[_sample_one_trajectory(dt, n_steps, tau_range, rng) for _ in range(n_trajectories)]
    )
    return np.stack(trajectories, axis=0), np.stack(tau_seqs, axis=0)
