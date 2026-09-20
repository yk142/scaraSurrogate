"""計算トルク法PTP制御を、真の物理モデル(オラクル)・グレーボックスサロゲート・
ブラックボックスサロゲートの3条件で比較する。

同一ゲイン(control.py)を使い、複数の初期条件(目標からのオフセット・初期角速度)で
最終追従誤差を比較する。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.control import PTPController, angular_error, run_ptp_surrogate, run_ptp_true
from src.model import AutoregressiveModel, GrayBoxModel, NSSModel
from src.train import CURRICULUM, DT, SEED, train

TARGET = np.array([0.5, -0.5])
N_STEPS = 1500  # 30秒
WINDOW = 50  # 最終1秒で追従誤差を評価
SUCCESS_THRESHOLD = 0.1

# 目標からのオフセット・初期角速度をずらした複数のIC
OFFSETS = [(-0.8, 0.6), (-0.5, 0.5), (-0.3, 0.3), (0.3, -0.3), (0.5, -0.5), (0.8, -0.6)]
VELOCITY_OFFSETS = [(0.0, 0.0), (0.5, -0.5)]

OUT_DIR = "outputs"


def make_test_ics() -> list[np.ndarray]:
    ics = []
    for dq1, dq2 in OFFSETS:
        for dv1, dv2 in VELOCITY_OFFSETS:
            ics.append(np.array([TARGET[0] + dq1, TARGET[1] + dq2, dv1, dv2]))
    return ics


def final_error(traj: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(angular_error(traj[-WINDOW:, :2], target)).sum(axis=-1).mean())


def evaluate_all(blackbox: AutoregressiveModel, graybox: AutoregressiveModel) -> dict[str, list[float]]:
    ics = make_test_ics()
    results = {"真の物理モデル": [], "グレーボックス": [], "ブラックボックス": []}
    for ic in ics:
        true_traj = run_ptp_true(ic, TARGET, N_STEPS, DT, PTPController())
        gray_traj = run_ptp_surrogate(graybox, ic, TARGET, N_STEPS, DT, PTPController())
        black_traj = run_ptp_surrogate(blackbox, ic, TARGET, N_STEPS, DT, PTPController())

        results["真の物理モデル"].append(final_error(true_traj, TARGET))
        results["グレーボックス"].append(final_error(gray_traj, TARGET))
        results["ブラックボックス"].append(final_error(black_traj, TARGET))
        print(
            f"IC={ic}: 真値={results['真の物理モデル'][-1]:.4f} "
            f"グレー={results['グレーボックス'][-1]:.4f} "
            f"黒={results['ブラックボックス'][-1]:.4f}"
        )
    return results


def plot_representative_trajectory(blackbox: AutoregressiveModel, graybox: AutoregressiveModel) -> None:
    ic = np.array([TARGET[0] - 0.8, TARGET[1] + 0.6, 0.0, 0.0])
    t = np.arange(N_STEPS + 1) * DT

    true_traj = run_ptp_true(ic, TARGET, N_STEPS, DT, PTPController())
    gray_traj = run_ptp_surrogate(graybox, ic, TARGET, N_STEPS, DT, PTPController())
    black_traj = run_ptp_surrogate(blackbox, ic, TARGET, N_STEPS, DT, PTPController())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, joint_label in enumerate(["theta1", "theta2"]):
        ax = axes[i]
        ax.plot(t, true_traj[:, i], label="真値", linewidth=1.5)
        ax.plot(t, gray_traj[:, i], "--", label="グレーボックス", linewidth=1.2)
        ax.plot(t, black_traj[:, i], "--", label="ブラックボックス", linewidth=1.2)
        ax.axhline(TARGET[i], color="gray", linestyle=":", linewidth=1, label="目標")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"{joint_label} [rad]")
        ax.set_title(joint_label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(f"PTP制御の代表試行 (IC offset: θ1-0.8, θ2+0.6)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ptp_trajectory.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/ptp_trajectory.png")


def plot_error_summary(results: dict[str, list[float]]) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.boxplot(list(results.values()), tick_labels=list(results.keys()))
    plt.axhline(SUCCESS_THRESHOLD, color="red", linestyle=":", linewidth=1, label="成功しきい値")
    plt.ylabel("最終1秒間の平均追従誤差 [rad]")
    plt.title(f"PTP制御の最終追従誤差 (n={len(next(iter(results.values())))} IC)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ptp_error_summary.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/ptp_error_summary.png")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    blackbox = train(curriculum=CURRICULUM, model_cls=NSSModel)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graybox = train(curriculum=CURRICULUM, model_cls=GrayBoxModel)

    results = evaluate_all(blackbox, graybox)
    plot_representative_trajectory(blackbox, graybox)
    plot_error_summary(results)

    print(f"\n=== 成功数 (final_err < {SUCCESS_THRESHOLD}) ===")
    for label, errs in results.items():
        n_success = sum(1 for e in errs if e < SUCCESS_THRESHOLD)
        print(f"{label}: {n_success}/{len(errs)}")
