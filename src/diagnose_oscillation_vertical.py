"""Phase2-M4 (#29): グレーボックスのPTP持続振動の原因を診断する。

Phase2-M3で最も成功率の低かったIC([0,0,0,0], 目標からのoffset=-0.5,+0.5)を
代表として、グレーボックス自身の閉ループ軌道を記録し、その軌道上で
「真の摩擦トルク」と「グレーボックスが実際に使った残差予測」を比較する。
振動が起きている区間で摩擦推定誤差が特に悪化しているかを確認する
(Phase 1のM5, diagnose_overshoot.py相当)。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.control_vertical import PTPControllerVertical
from src.evaluate_ptp_vertical import TARGET
from src.model_vertical import GrayBoxModelVertical
from src.physics import mass_matrix
from src.physics_vertical import friction_torque
from src.train_vertical import CURRICULUM, DT, SEED, train

IC = np.array([0.0, 0.0, 0.0, 0.0])
N_STEPS = 1500  # 30秒
OUT_DIR = "outputs"


def run_graybox_closed_loop(graybox, initial_state, target, n_steps, dt):
    controller = PTPControllerVertical()
    controller.reset()
    state = initial_state.copy()
    states = [state.copy()]
    taus = []
    for _ in range(n_steps):
        tau = controller.compute(state, target, dt)
        taus.append(tau.copy())
        state = graybox.rollout(state, 1, tau_seq=tau[None, :])[-1]
        states.append(state.copy())
    return np.stack(states, axis=0), np.stack(taus, axis=0)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graybox = train(curriculum=CURRICULUM, model_cls=GrayBoxModelVertical)

    states, taus = run_graybox_closed_loop(graybox, IC, TARGET, N_STEPS, DT)
    q2 = states[:-1, 1]
    q_dot = states[:-1, 2:]

    # 真の摩擦が加速度に与える影響 = -M(q)^-1 @ friction_torque(q_dot)
    # (Phase2-M7で残差ネットの出力をトルクから加速度に変更したため、比較対象も
    # 加速度領域に揃える)
    true_friction_torque = friction_torque(q_dot)
    M = mass_matrix(q2)
    true_friction_accel = -np.linalg.solve(M, true_friction_torque[..., None])[..., 0]

    with torch.no_grad():
        state_t = torch.as_tensor(states[:-1], dtype=torch.float32)
        tau_t = torch.as_tensor(taus, dtype=torch.float32)
        pred_residual = graybox.residual_accel(state_t, tau_t).numpy()
    residual_error = pred_residual - true_friction_accel

    t = np.arange(N_STEPS) * DT
    t_state = np.arange(N_STEPS + 1) * DT

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, joint_label in enumerate(["theta1", "theta2"]):
        ax = axes[i]
        ax.plot(t_state, states[:, i], label="グレーボックス自身の閉ループ軌道")
        ax.axhline(TARGET[i], color="gray", linestyle=":", linewidth=1, label="目標")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"{joint_label} [rad]")
        ax.set_title(joint_label, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("グレーボックス自身の閉ループ軌道(IC=[0,0,0,0])")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/vertical_oscillation_trajectory.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/vertical_oscillation_trajectory.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, joint_label in enumerate(["joint1", "joint2"]):
        ax = axes[i]
        ax.plot(t, true_friction_accel[:, i], label="真の摩擦による加速度", linewidth=1.2)
        ax.plot(t, pred_residual[:, i], "--", label="グレーボックスの残差予測", linewidth=1.0)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("accel [rad/s^2]")
        ax.set_title(joint_label, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("グレーボックス自身の軌道上での摩擦推定(加速度領域、真値 vs 実際に使った予測)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/vertical_oscillation_friction.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/vertical_oscillation_friction.png")

    print(f"\n摩擦推定誤差RMS(全期間): joint1={np.sqrt((residual_error[:, 0]**2).mean()):.4f} "
          f"joint2={np.sqrt((residual_error[:, 1]**2).mean()):.4f}")
    print(f"角速度レンジ: q1_dot=[{q_dot[:, 0].min():.3f}, {q_dot[:, 0].max():.3f}] "
          f"q2_dot=[{q_dot[:, 1].min():.3f}, {q_dot[:, 1].max():.3f}]")


if __name__ == "__main__":
    main()
