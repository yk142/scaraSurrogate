"""Phase3-M3 (#54): 3D空間マニピュレータで計算トルク法PTP制御を、真の物理モデル
(オラクル)・グレーボックスサロゲート・ブラックボックスサロゲートの3条件で
比較する(Phase 1/2のevaluate_ptp.py相当)。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.control import angular_error
from src.control_3d import PTPController3D, run_ptp_surrogate, run_ptp_true
from src.model_3d import AutoregressiveModel3D, NSSModel3D, StructuredFrictionGrayBoxModel3D
from src.train_3d import CURRICULUM, DT, SEED, train

TARGET = np.array([0.5, 0.5, -0.5])
N_STEPS = 1500  # 30秒
WINDOW = 50
SUCCESS_THRESHOLD = 0.1

# Phase2のOFFSETS(q1,q2)にヨーオフセットを組み合わせ、12ICにする。
OFFSETS = [(-0.8, 0.6), (-0.5, 0.5), (-0.3, 0.3), (0.3, -0.3), (0.5, -0.5), (0.8, -0.6)]
YAW_OFFSETS = [-1.0, 1.0]

OUT_DIR = "outputs"


def make_test_ics() -> list[np.ndarray]:
    ics = []
    for dq1, dq2 in OFFSETS:
        for dq0 in YAW_OFFSETS:
            ics.append(np.array([TARGET[0] + dq0, TARGET[1] + dq1, TARGET[2] + dq2, 0.0, 0.0, 0.0]))
    return ics


def final_error(traj: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(angular_error(traj[-WINDOW:, :3], target)).sum(axis=-1).mean())


def evaluate_all(blackbox: AutoregressiveModel3D, graybox: AutoregressiveModel3D) -> dict[str, list[float]]:
    ics = make_test_ics()
    results = {"真の物理モデル": [], "グレーボックス": [], "ブラックボックス": []}
    for ic in ics:
        true_traj = run_ptp_true(ic, TARGET, N_STEPS, DT, PTPController3D())
        gray_traj = run_ptp_surrogate(graybox, ic, TARGET, N_STEPS, DT, PTPController3D())
        black_traj = run_ptp_surrogate(blackbox, ic, TARGET, N_STEPS, DT, PTPController3D())

        results["真の物理モデル"].append(final_error(true_traj, TARGET))
        results["グレーボックス"].append(final_error(gray_traj, TARGET))
        results["ブラックボックス"].append(final_error(black_traj, TARGET))
        print(
            f"IC={ic}: 真値={results['真の物理モデル'][-1]:.4f} "
            f"グレー={results['グレーボックス'][-1]:.4f} "
            f"黒={results['ブラックボックス'][-1]:.4f}"
        )
    return results


def plot_representative_trajectory(blackbox: AutoregressiveModel3D, graybox: AutoregressiveModel3D) -> None:
    ic = np.array([TARGET[0] - 1.0, TARGET[1] - 0.8, TARGET[2] + 0.6, 0.0, 0.0, 0.0])
    t = np.arange(N_STEPS + 1) * DT

    true_traj = run_ptp_true(ic, TARGET, N_STEPS, DT, PTPController3D())
    gray_traj = run_ptp_surrogate(graybox, ic, TARGET, N_STEPS, DT, PTPController3D())
    black_traj = run_ptp_surrogate(blackbox, ic, TARGET, N_STEPS, DT, PTPController3D())

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for i, joint_label in enumerate(["q0(ヨー)", "q1(肩)", "q2(肘)"]):
        ax = axes[i]
        ax.plot(t, true_traj[:, i], label="真値", linewidth=1.5)
        ax.plot(t, gray_traj[:, i], "--", label="グレーボックス", linewidth=1.2)
        ax.plot(t, black_traj[:, i], "--", label="ブラックボックス", linewidth=1.2)
        ax.axhline(TARGET[i], color="gray", linestyle=":", linewidth=1, label="目標")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"{joint_label} [rad]")
        ax.set_title(joint_label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Phase3: PTP制御の代表試行")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/yaw_ptp_trajectory.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/yaw_ptp_trajectory.png")


def plot_error_summary(results: dict[str, list[float]]) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.boxplot(list(results.values()), tick_labels=list(results.keys()))
    plt.axhline(SUCCESS_THRESHOLD, color="red", linestyle=":", linewidth=1, label="成功しきい値")
    plt.ylabel("最終1秒間の平均追従誤差 [rad]")
    plt.title(f"Phase3: PTP制御の最終追従誤差 (n={len(next(iter(results.values())))} IC)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/yaw_ptp_error_summary.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/yaw_ptp_error_summary.png")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    blackbox = train(curriculum=CURRICULUM, model_cls=NSSModel3D)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graybox = train(curriculum=CURRICULUM, model_cls=StructuredFrictionGrayBoxModel3D)

    results = evaluate_all(blackbox, graybox)
    plot_representative_trajectory(blackbox, graybox)
    plot_error_summary(results)

    print(f"\n=== 成功数 (final_err < {SUCCESS_THRESHOLD}) ===")
    for label, errs in results.items():
        n_success = sum(1 for e in errs if e < SUCCESS_THRESHOLD)
        print(f"{label}: {n_success}/{len(errs)}")
