"""Phase2-M9 (#38): 垂直面アームでPTP制御の過渡応答(整定時間・軌道誤差の
時間変化)を真の物理モデルとグレーボックスサロゲートで比較する。
Phase 1のM7 (`src/evaluate_transient.py`) 相当。

Phase2-M8で最終誤差の到達判定(PTP成功数12/12)は達成できたが、それだけでは
過渡応答全体の一致度は検証できていない。整定時間そのものと、軌道誤差の
時間変化を確認する。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.control_vertical import PTPControllerVertical, run_ptp_surrogate, run_ptp_true
from src.evaluate_ptp_vertical import N_STEPS, SUCCESS_THRESHOLD, TARGET, make_test_ics
from src.evaluate_transient import settling_time, trajectory_error
from src.model_vertical import StructuredFrictionGrayBoxModelVertical
from src.train_vertical import CURRICULUM, DT, SEED, train

OUT_DIR = "outputs"


def evaluate_settling_times(graybox) -> tuple[list[str], list[float], list[float]]:
    ics = make_test_ics()
    labels, true_times, gray_times = [], [], []
    for ic in ics:
        true_traj = run_ptp_true(ic, TARGET, N_STEPS, DT, PTPControllerVertical())
        gray_traj = run_ptp_surrogate(graybox, ic, TARGET, N_STEPS, DT, PTPControllerVertical())
        labels.append(f"({ic[0]-TARGET[0]:+.1f},{ic[1]-TARGET[1]:+.1f})")
        true_times.append(settling_time(true_traj, TARGET, DT))
        gray_times.append(settling_time(gray_traj, TARGET, DT))
        print(f"IC offset={labels[-1]}: 真値整定={true_times[-1]:.2f}s グレー整定={gray_times[-1]:.2f}s")
    return labels, true_times, gray_times


def plot_settling_times(labels: list[str], true_times: list[float], gray_times: list[float]) -> None:
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 4.5))
    plt.bar(x - width / 2, true_times, width, label="真の物理モデル")
    plt.bar(x + width / 2, gray_times, width, label="グレーボックス")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.xlabel("IC offset (θ1, θ2)")
    plt.ylabel("整定時間 [s]")
    plt.title(f"Phase2: 整定時間の比較(しきい値={SUCCESS_THRESHOLD}, n={len(labels)} IC)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/vertical_transient_settling_time.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/vertical_transient_settling_time.png")


def plot_trajectory_error_curves(graybox) -> None:
    ics = make_test_ics()
    errors = []
    for ic in ics:
        true_traj = run_ptp_true(ic, TARGET, N_STEPS, DT, PTPControllerVertical())
        gray_traj = run_ptp_surrogate(graybox, ic, TARGET, N_STEPS, DT, PTPControllerVertical())
        errors.append(trajectory_error(true_traj, gray_traj))
    errors = np.stack(errors, axis=0)

    t = np.arange(N_STEPS + 1) * DT
    mean_err = errors.mean(axis=0)
    min_err = errors.min(axis=0)
    max_err = errors.max(axis=0)

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, mean_err, label="平均(真値-グレーボックス誤差)")
    plt.fill_between(t, min_err, max_err, alpha=0.2, label="12IC間のレンジ")
    plt.xlabel("time [s]")
    plt.ylabel("角度誤差(joint1+joint2) [rad]")
    plt.title("Phase2: PTP軌道の真値-グレーボックス間誤差(過渡応答全体)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/vertical_transient_trajectory_error.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/vertical_transient_trajectory_error.png")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graybox = train(curriculum=CURRICULUM, model_cls=StructuredFrictionGrayBoxModelVertical)

    labels, true_times, gray_times = evaluate_settling_times(graybox)
    plot_settling_times(labels, true_times, gray_times)
    plot_trajectory_error_curves(graybox)

    diffs = [abs(g - t) for t, g in zip(true_times, gray_times) if not (np.isnan(t) or np.isnan(g))]
    print("\n=== 整定時間差の統計 ===")
    print(f"平均差: {np.mean(diffs):.3f}s  最大差: {np.max(diffs):.3f}s")
