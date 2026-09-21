"""Phase2: 重力あり垂直面2関節アーム用の学習・評価用データセット生成。

IC・トルクのサンプリング関数はシステムに依存しないため`src/dataset.py`の
ものをそのまま再利用し、シミュレータのみ`physics_vertical.simulate`に差し替える。

TAU_RANGE: 重力トルクの最大値(joint1: (m1*lc1+m2*l1+m2*lc2)*g≈9.81,
joint2: m2*lc2*g≈2.45)を上回るように、Phase 1の(-1,1)から拡大する。
"""
import numpy as np

from src.dataset import sample_initial_states, sample_torque_sequence
from src.physics_vertical import simulate

TAU_RANGE = (-10.0, 10.0)

# 重力あり2重振り子はカオス系で、IC・トルクを素朴に一様サンプリングすると
# ごく一部の組み合わせで角速度が数十rad/s(実機ではあり得ない領域)まで発散する
# ことがある(Phase2-M2の検証で発見)。実機の関節速度制限を模した安全弁として、
# この上限を超える軌道は棄却して引き直す。
VELOCITY_LIMIT = 10.0
MAX_RESAMPLE_ATTEMPTS = 20


def _sample_one_trajectory(dt: float, n_steps: int, tau_range: tuple[float, float], rng: np.random.Generator):
    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        ic = sample_initial_states(1, rng)[0]
        tau_seq = sample_torque_sequence(n_steps, rng, tau_range=tau_range)
        traj = simulate(ic, dt, n_steps, tau=tau_seq)
        if np.abs(traj[:, 2:]).max() <= VELOCITY_LIMIT:
            return traj, tau_seq
    # 上限回数リサンプルしても収まらなければ、そのまま採用する(稀なケース)。
    return traj, tau_seq


def generate_controlled_trajectories(
    n_trajectories: int, dt: float, n_steps: int, seed: int, tau_range: tuple[float, float] = TAU_RANGE
) -> tuple[np.ndarray, np.ndarray]:
    """ランダムトルク列で駆動した軌道データセットを生成する(重力あり)。

    角速度が`VELOCITY_LIMIT`を超える(実機ではあり得ない)軌道は棄却して
    引き直す(`_sample_one_trajectory`参照)。

    Returns:
        trajectories: shape (n_traj, n_steps + 1, 4)
        tau_seqs:     shape (n_traj, n_steps, 2)
    """
    rng = np.random.default_rng(seed)
    trajectories, tau_seqs = zip(
        *[_sample_one_trajectory(dt, n_steps, tau_range, rng) for _ in range(n_trajectories)]
    )
    return np.stack(trajectories, axis=0), np.stack(tau_seqs, axis=0)
