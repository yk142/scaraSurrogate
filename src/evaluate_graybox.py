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


def plot_theta_trajectory(
    blackbox: AutoregressiveModel, graybox: AutoregressiveModel, conditions: dict[str, np.ndarray]
) -> None:
    """代表IC1点でのtheta1(t), theta2(t)を真値・ブラックボックス・グレーボックスで比較する。"""
    rng = np.random.default_rng(SEED_TEST)
    ic = sample_initial_states(1, rng)[0]
    t = np.arange(N_STEPS_EVAL + 1) * DT

    fig, axes = plt.subplots(2, len(conditions), figsize=(6 * len(conditions), 8), squeeze=False)
    for col, (label, tau_seq) in enumerate(conditions.items()):
        true_traj = true_rollout(ic[None, :], DT, N_STEPS_EVAL, tau_seq=tau_seq)[0]
        blackbox_traj = blackbox.rollout(ic, N_STEPS_EVAL, tau_seq=tau_seq)
        graybox_traj = graybox.rollout(ic, N_STEPS_EVAL, tau_seq=tau_seq)

        for row, joint_label in enumerate(["theta1", "theta2"]):
            ax = axes[row, col]
            ax.plot(t, np.unwrap(true_traj[:, row]), label="真値", linewidth=1.5)
            ax.plot(t, np.unwrap(blackbox_traj[:, row]), "--", label="ブラックボックス", linewidth=1.2)
            ax.plot(t, np.unwrap(graybox_traj[:, row]), "--", label="グレーボックス", linewidth=1.2)
            ax.axvline(CURRICULUM[-1][0] * DT, color="gray", linestyle=":", linewidth=1)
            ax.set_xlabel("time [s]")
            ax.set_ylabel(f"{joint_label} [rad]")
            ax.set_title(f"{label} ({joint_label})", fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle(f"角度の時系列比較 (IC: θ1={ic[0]:.2f}, θ2={ic[1]:.2f}, θ̇1={ic[2]:.2f}, θ̇2={ic[3]:.2f})")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/theta_trajectory.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/theta_trajectory.png")


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
    plot_theta_trajectory(blackbox, graybox, conditions)

    print(f"\n=== 開ループロールアウトRMSE (t={N_STEPS_EVAL*DT:.1f}s) ===")
    for label in blackbox_curves:
        print(
            f"{label}: ブラックボックス={blackbox_curves[label][-1]:.4f} "
            f"グレーボックス={graybox_curves[label][-1]:.4f}"
        )
