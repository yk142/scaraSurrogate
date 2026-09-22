"""Phase3-M2 (#52): 3D空間マニピュレータでブラックボックス版とグレーボックス版の
開ループロールアウト誤差を比較する(Phase 1/2のM2相当)。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.dataset_3d import TAU_RANGE, sample_initial_states
from src.model_3d import AutoregressiveModel3D, NSSModel3D, StructuredFrictionGrayBoxModel3D
from src.physics_3d import simulate as simulate_3d
from src.rollout import true_rollout
from src.train_3d import CURRICULUM, DT, SEED, train

N_STEPS_EVAL = 300  # 6秒分
N_TEST_TRAJ = 200
SEED_TEST = 1000
OUT_DIR = "outputs"


def rmse_curve_3d(true_traj: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
    """horizon stepごとのRMSEを返す(角度3次元はwrap済み差、角速度3次元は単純差)。

    Phase1/2の`rollout.rmse_curve`は2関節(角度2+角速度2)専用のため、
    3関節(角度3+角速度3)用にここで定義する。

    true_traj, pred_traj: shape (n_steps+1, n_ic, 6) -> Returns: shape (n_steps+1,)
    """
    angle_err = np.arctan2(
        np.sin(true_traj[..., :3] - pred_traj[..., :3]),
        np.cos(true_traj[..., :3] - pred_traj[..., :3]),
    )
    vel_err = true_traj[..., 3:] - pred_traj[..., 3:]
    sq_err = (angle_err**2).sum(axis=-1) + (vel_err**2).sum(axis=-1)
    return np.sqrt(sq_err.mean(axis=1))


def make_control_conditions() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED_TEST + 1)
    n_holds = int(np.ceil(N_STEPS_EVAL / 5))
    hold_values = rng.uniform(*TAU_RANGE, size=(n_holds, 3))
    random_tau = np.repeat(hold_values, 5, axis=0)[:N_STEPS_EVAL]
    return {
        "ゼロトルク": np.zeros((N_STEPS_EVAL, 3)),
        "ランダムトルク列": random_tau,
    }


def rmse_at_horizon(model: AutoregressiveModel3D, conditions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED_TEST)
    ics = sample_initial_states(N_TEST_TRAJ, rng)

    curves = {}
    for label, tau_seq in conditions.items():
        true_traj = np.transpose(
            true_rollout(ics, DT, N_STEPS_EVAL, tau_seq=tau_seq, simulate_fn=simulate_3d), (1, 0, 2)
        )
        pred_traj = model.rollout(ics, N_STEPS_EVAL, tau_seq=tau_seq)
        curves[label] = rmse_curve_3d(true_traj, pred_traj)
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

    fig.suptitle("Phase3: ブラックボックス vs グレーボックス 開ループロールアウト誤差")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/yaw_graybox_comparison.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/yaw_graybox_comparison.png")


def plot_theta_trajectory(
    blackbox: AutoregressiveModel3D, graybox: AutoregressiveModel3D, conditions: dict[str, np.ndarray]
) -> None:
    """代表IC1点でのq0(t),q1(t),q2(t)を真値・ブラックボックス・グレーボックスで比較する
    (Phase 1のevaluate_graybox.plot_theta_trajectory相当)。
    """
    rng = np.random.default_rng(SEED_TEST)
    ic = sample_initial_states(1, rng)[0]
    t = np.arange(N_STEPS_EVAL + 1) * DT

    fig, axes = plt.subplots(3, len(conditions), figsize=(6 * len(conditions), 11), squeeze=False)
    for col, (label, tau_seq) in enumerate(conditions.items()):
        true_traj = simulate_3d(ic, DT, N_STEPS_EVAL, tau=tau_seq)
        blackbox_traj = blackbox.rollout(ic, N_STEPS_EVAL, tau_seq=tau_seq)
        graybox_traj = graybox.rollout(ic, N_STEPS_EVAL, tau_seq=tau_seq)

        for row, joint_label in enumerate(["q0(ヨー)", "q1(肩)", "q2(肘)"]):
            ax = axes[row, col]
            ax.plot(t, np.unwrap(true_traj[:, row]), label="真値", linewidth=1.5)
            ax.plot(t, np.unwrap(blackbox_traj[:, row]), "--", label="ブラックボックス", linewidth=1.2)
            ax.plot(t, np.unwrap(graybox_traj[:, row]), "--", label="グレーボックス", linewidth=1.2)
            ax.axvline(CURRICULUM[-1][0] * DT, color="gray", linestyle=":", linewidth=1)
            ax.set_xlabel("time [s]")
            ax.set_ylabel(f"{joint_label} [rad]")
            ax.set_title(f"{label} ({joint_label})", fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle(
        f"Phase3: 角度の時系列比較 (IC: q0={ic[0]:.2f}, q1={ic[1]:.2f}, q2={ic[2]:.2f}, "
        f"q0̇={ic[3]:.2f}, q1̇={ic[4]:.2f}, q2̇={ic[5]:.2f})"
    )
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/yaw_theta_trajectory.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/yaw_theta_trajectory.png")


if __name__ == "__main__":
    conditions = make_control_conditions()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    blackbox = train(curriculum=CURRICULUM, model_cls=NSSModel3D)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graybox = train(curriculum=CURRICULUM, model_cls=StructuredFrictionGrayBoxModel3D)

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
