"""M5 (#9): グレーボックスのPTPオーバーシュート原因調査。

代表IC(offset θ1-0.8, θ2+0.6)の真の物理モデルによるPTP閉ループ軌道を追跡しながら、
各時刻の(state, tau)についてグレーボックスの残差ネット `residual_torque` が予測する
摩擦トルクと、真の摩擦トルク `physics.friction_torque` を直接比較する。

これにより、オーバーシュートが起きるt≈2〜6秒の区間で残差予測誤差が特に大きく
なっているか(仮説1: 学習データ分布内での精度不足)、それとも残差予測はほぼ正確
なのにコリオリ結合を介して誤差が伝播しているのか(仮説2)を切り分ける。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.control import PTPController, run_ptp_true
from src.dataset import THETA_DOT_RANGE
from src.model import GrayBoxModel
from src.physics import friction_torque
from src.train import CURRICULUM, DT, SEED, train

TARGET = np.array([0.5, -0.5])
IC = np.array([TARGET[0] - 0.8, TARGET[1] + 0.6, 0.0, 0.0])
N_STEPS = 1500  # 30秒

OUT_DIR = "outputs"


def run_true_with_tau_log(initial_state: np.ndarray, target: np.ndarray, n_steps: int, dt: float):
    """run_ptp_true と同じ閉ループ制御だが、各ステップのtauも記録して返す。"""
    from src.physics import rk4_step

    controller = PTPController()
    controller.reset()
    state = initial_state.copy()
    states = [state.copy()]
    taus = []
    for _ in range(n_steps):
        tau = controller.compute(state, target, dt)
        taus.append(tau.copy())
        state = rk4_step(state, dt, tau)
        states.append(state.copy())
    return np.stack(states, axis=0), np.stack(taus, axis=0)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graybox = train(curriculum=CURRICULUM, model_cls=GrayBoxModel)

    states, taus = run_true_with_tau_log(IC, TARGET, N_STEPS, DT)
    q_dot = states[:-1, 2:]  # tauが適用された時点の状態(区分定数トルクの開始点)

    true_friction = friction_torque(q_dot)

    with torch.no_grad():
        state_t = torch.as_tensor(states[:-1], dtype=torch.float32)
        tau_t = torch.as_tensor(taus, dtype=torch.float32)
        pred_residual = graybox.residual_torque(state_t, tau_t).numpy()

    residual_error = pred_residual - true_friction
    t = np.arange(N_STEPS) * DT

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, joint_label in enumerate(["joint1", "joint2"]):
        ax = axes[i]
        ax.plot(t, true_friction[:, i], label="真の摩擦トルク", linewidth=1.5)
        ax.plot(t, pred_residual[:, i], "--", label="グレーボックスの残差予測", linewidth=1.2)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("torque [Nm]")
        ax.set_title(joint_label, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("真の摩擦トルク vs グレーボックス残差予測(真値軌道上)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/overshoot_residual_comparison.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/overshoot_residual_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for i, joint_label in enumerate(["joint1", "joint2"]):
        ax = axes[i]
        ax.plot(t, residual_error[:, i], label="残差予測誤差", color="tab:red")
        ax.axvspan(2, 6, color="gray", alpha=0.15, label="オーバーシュート区間(目視)")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("torque error [Nm]")
        ax.set_title(joint_label, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("残差予測誤差(グレーボックス予測 - 真値)の時間変化")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/overshoot_residual_error.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/overshoot_residual_error.png")

    window = (t >= 2) & (t <= 6)
    print("\n=== 摩擦推定誤差の統計 (RMS) ===")
    print(f"全区間(0-30s):    joint1={np.sqrt((residual_error[:, 0]**2).mean()):.4f} "
          f"joint2={np.sqrt((residual_error[:, 1]**2).mean()):.4f}")
    print(f"オーバーシュート区間(2-6s): joint1={np.sqrt((residual_error[window, 0]**2).mean()):.4f} "
          f"joint2={np.sqrt((residual_error[window, 1]**2).mean()):.4f}")

    q_dot_window = q_dot[window]
    print(f"\n学習データの角速度レンジ: {THETA_DOT_RANGE}")
    print(f"オーバーシュート区間中の角速度レンジ: "
          f"joint1=[{q_dot_window[:, 0].min():.3f}, {q_dot_window[:, 0].max():.3f}] "
          f"joint2=[{q_dot_window[:, 1].min():.3f}, {q_dot_window[:, 1].max():.3f}]")


if __name__ == "__main__":
    main()
