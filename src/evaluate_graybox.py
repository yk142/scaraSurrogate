"""ブラックボックス版とグレーボックス版の開ループロールアウト誤差を比較する。

同一の学習レシピ(マルチステップ損失+カリキュラム学習)で両アーキテクチャを学習し、
アーキテクチャの違いだけを切り分けて比較する(振り子プロジェクトのM14相当)。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.dataset import sample_initial_states, sample_torque_sequence
from src.model import AutoregressiveModel, GrayBoxModel, NSSModel
from src.rollout import rmse_curve, true_rollout
from src.train import CURRICULUM, DT, SEED, train

N_STEPS_EVAL = 300  # 6秒分
N_TEST_TRAJ = 200
SEED_TEST = 1000
OUT_DIR = "outputs"


def make_control_conditions() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED_TEST + 1)
    return {
        "ゼロトルク": np.zeros((N_STEPS_EVAL, 2)),
        "ランダムトルク列": sample_torque_sequence(N_STEPS_EVAL, rng),
    }


def rmse_at_horizon(model: AutoregressiveModel, conditions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED_TEST)
    ics = sample_initial_states(N_TEST_TRAJ, rng)

    curves = {}
    for label, tau_seq in conditions.items():
        true_traj = np.transpose(true_rollout(ics, DT, N_STEPS_EVAL, tau_seq=tau_seq), (1, 0, 2))
        pred_traj = model.rollout(ics, N_STEPS_EVAL, tau_seq=tau_seq)
        curves[label] = rmse_curve(true_traj, pred_traj)
    return curves


def plot_comparison(blackbox_curves: dict[str, np.ndarray], graybox_curves: dict[str, np.ndarray]) -> None:
    t = np.arange(N_STEPS_EVAL + 1) * DT
    fig, axes = plt.subplots(1, len(blackbox_curves), figsize=(6 * len(blackbox_curves), 4.5))
    if len(blackbox_curves) == 1:
        axes = [axes]
    for ax, label in zip(axes, blackbox_curves):
        ax.plot(t, blackbox_curves[label], label="ブラックボックス")
        ax.plot(t, graybox_curves[label], label="グレーボックス")
        ax.axvline(CURRICULUM[-1][0] * DT, color="gray", linestyle="--", linewidth=1, label="学習ホライズン境界")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("RMSE")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("ブラックボックス vs グレーボックス: 開ループロールアウト誤差")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graybox_comparison.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/graybox_comparison.png")


if __name__ == "__main__":
    conditions = make_control_conditions()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    blackbox = train(curriculum=CURRICULUM, model_cls=NSSModel)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graybox = train(curriculum=CURRICULUM, model_cls=GrayBoxModel)

    blackbox_curves = rmse_at_horizon(blackbox, conditions)
    graybox_curves = rmse_at_horizon(graybox, conditions)

    plot_comparison(blackbox_curves, graybox_curves)

    print(f"\n=== 開ループロールアウトRMSE (t={N_STEPS_EVAL*DT:.1f}s) ===")
    for label in blackbox_curves:
        print(
            f"{label}: ブラックボックス={blackbox_curves[label][-1]:.4f} "
            f"グレーボックス={graybox_curves[label][-1]:.4f}"
        )
