"""Phase2-M2 (#25): 垂直面アームでブラックボックス版とグレーボックス版の
開ループロールアウト誤差を比較する(Phase 1のM2, evaluate_graybox.py相当)。

ブラックボックスはシステムに依存しない`src.model.NSSModel`をそのまま再利用し、
グレーボックスは重力項G(q)・摩擦の関数形をハードコードした
`src.model_vertical.StructuredFrictionGrayBoxModelVertical`を用いる(Phase2-M8, #36)。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.dataset import sample_initial_states
from src.dataset_vertical import TAU_RANGE, generate_controlled_trajectories
from src.model import AutoregressiveModel, NSSModel
from src.model_vertical import StructuredFrictionGrayBoxModelVertical
from src.physics_vertical import simulate as simulate_vertical
from src.rollout import rmse_curve, true_rollout
from src.train_vertical import CURRICULUM, DT, SEED, train

N_STEPS_EVAL = 300  # 6秒分
N_TEST_TRAJ = 200
SEED_TEST = 1000
OUT_DIR = "outputs"


def make_control_conditions() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED_TEST + 1)
    n_holds = int(np.ceil(N_STEPS_EVAL / 5))
    hold_values = rng.uniform(*TAU_RANGE, size=(n_holds, 2))
    random_tau = np.repeat(hold_values, 5, axis=0)[:N_STEPS_EVAL]
    return {
        "ゼロトルク": np.zeros((N_STEPS_EVAL, 2)),
        "ランダムトルク列": random_tau,
    }


def rmse_at_horizon(model: AutoregressiveModel, conditions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED_TEST)
    ics = sample_initial_states(N_TEST_TRAJ, rng)

    curves = {}
    for label, tau_seq in conditions.items():
        true_traj = np.transpose(
            true_rollout(ics, DT, N_STEPS_EVAL, tau_seq=tau_seq, simulate_fn=simulate_vertical), (1, 0, 2)
        )
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

    fig.suptitle("Phase2: ブラックボックス vs グレーボックス(重力あり)開ループロールアウト誤差")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/vertical_graybox_comparison.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/vertical_graybox_comparison.png")


if __name__ == "__main__":
    conditions = make_control_conditions()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    blackbox = train(curriculum=CURRICULUM, model_cls=NSSModel)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graybox = train(curriculum=CURRICULUM, model_cls=StructuredFrictionGrayBoxModelVertical)

    blackbox_curves = rmse_at_horizon(blackbox, conditions)
    graybox_curves = rmse_at_horizon(graybox, conditions)

    plot_comparison(blackbox_curves, graybox_curves)

    print(f"\n=== 開ループロールアウトRMSE (t={N_STEPS_EVAL*DT:.1f}s) ===")
    for label in blackbox_curves:
        print(
            f"{label}: ブラックボックス={blackbox_curves[label][-1]:.4f} "
            f"グレーボックス={graybox_curves[label][-1]:.4f}"
        )
