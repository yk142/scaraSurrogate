"""M9 (#17): 大オフセットIC(トルク飽和領域)での整定遅延を診断する。

M8で、IC offset=(+0.8,-0.6, 初期角速度あり)においてグレーボックスの整定時間が
真値より大幅に遅れることが判明した。このICでの真値・グレーボックス双方の
閉ループ軌道とトルク時系列を可視化し、トルク飽和の継続時間・振動の有無・
軌道が乖離し始めるタイミングを診断する。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.control import PTPController, TAU_MAX, angular_error
from src.evaluate_ptp import TARGET
from src.model import GrayBoxModel
from src.physics import friction_torque, rk4_step
from src.train import CURRICULUM, DT, SEED, train

IC = np.array([TARGET[0] + 0.8, TARGET[1] - 0.6, 0.5, -0.5])
N_STEPS = 1500  # 30秒
OUT_DIR = "outputs"


def run_with_tau_log(step_fn, initial_state: np.ndarray, target: np.ndarray, n_steps: int, dt: float):
    controller = PTPController()
    controller.reset()
    state = initial_state.copy()
    states = [state.copy()]
    taus = []
    for _ in range(n_steps):
        tau = controller.compute(state, target, dt)
        taus.append(tau.copy())
        state = step_fn(state, tau)
        states.append(state.copy())
    return np.stack(states, axis=0), np.stack(taus, axis=0)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graybox = train(curriculum=CURRICULUM, model_cls=GrayBoxModel)

    def true_step(state, tau):
        return rk4_step(state, DT, tau)

    def gray_step(state, tau):
        return graybox.rollout(state, 1, tau_seq=tau[None, :])[-1]

    true_states, true_taus = run_with_tau_log(true_step, IC, TARGET, N_STEPS, DT)
    gray_states, gray_taus = run_with_tau_log(gray_step, IC, TARGET, N_STEPS, DT)

    t = np.arange(N_STEPS + 1) * DT
    t_tau = np.arange(N_STEPS) * DT

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, joint_label in enumerate(["theta1", "theta2"]):
        ax = axes[0, i]
        ax.plot(t, true_states[:, i], label="真値", linewidth=1.5)
        ax.plot(t, gray_states[:, i], "--", label="グレーボックス", linewidth=1.2)
        ax.axhline(TARGET[i], color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"{joint_label} [rad]")
        ax.set_title(joint_label, fontsize=10)
        ax.legend(fontsize=8)

    for i, joint_label in enumerate(["tau1", "tau2"]):
        ax = axes[1, i]
        ax.plot(t_tau, true_taus[:, i], label="真値", linewidth=1.2)
        ax.plot(t_tau, gray_taus[:, i], "--", label="グレーボックス", linewidth=1.0)
        ax.axhline(TAU_MAX, color="red", linestyle=":", linewidth=1, label="TAU_MAX")
        ax.axhline(-TAU_MAX, color="red", linestyle=":", linewidth=1)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"{joint_label} [Nm]")
        ax.set_title(joint_label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(f"大オフセットIC(offset=+0.8,-0.6, 初期角速度あり)の軌道・トルク診断")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/saturation_diagnosis.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/saturation_diagnosis.png")

    true_sat_frac = (np.abs(true_taus) >= TAU_MAX - 1e-6).mean(axis=0)
    gray_sat_frac = (np.abs(gray_taus) >= TAU_MAX - 1e-6).mean(axis=0)
    print(f"\n真値のトルク飽和割合(全期間): tau1={true_sat_frac[0]:.3f} tau2={true_sat_frac[1]:.3f}")
    print(f"グレーのトルク飽和割合(全期間): tau1={gray_sat_frac[0]:.3f} tau2={gray_sat_frac[1]:.3f}")

    # 最初にトルク飽和が解除される時刻
    for label, taus in [("真値", true_taus), ("グレーボックス", gray_taus)]:
        sat = np.abs(taus).max(axis=-1) >= TAU_MAX - 1e-6
        if sat.any() and not sat.all():
            release_idx = np.where(~sat)[0][0]
            print(f"{label}: 最初にトルク飽和が解除される時刻 = {release_idx * DT:.2f}s")
        elif sat.all():
            print(f"{label}: 全期間トルク飽和")
        else:
            print(f"{label}: トルク飽和なし")

    err = np.abs(angular_error(true_states[:, :2], gray_states[:, :2])).sum(axis=-1)
    peak_idx = np.argmax(err)
    print(f"\n真値-グレー間の軌道誤差ピーク: t={peak_idx * DT:.2f}s, error={err[peak_idx]:.3f}rad")

    # 真値軌道上での摩擦推定誤差(このICでは角速度が大きい領域を通るため、
    # M5/M6/M8で見た低速域のバイアスとは別に、高速域での誤差がないか確認する)
    q_dot = true_states[:-1, 2:]
    true_friction = friction_torque(q_dot)
    with torch.no_grad():
        state_t = torch.as_tensor(true_states[:-1], dtype=torch.float32)
        tau_t = torch.as_tensor(true_taus, dtype=torch.float32)
        pred_residual = graybox.residual_torque(state_t, tau_t).numpy()
    residual_error = pred_residual - true_friction

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, joint_label in enumerate(["joint1", "joint2"]):
        ax = axes[i]
        ax.plot(t_tau, true_friction[:, i], label="真の摩擦トルク", linewidth=1.5)
        ax.plot(t_tau, pred_residual[:, i], "--", label="グレーボックスの残差予測", linewidth=1.2)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("torque [Nm]")
        ax.set_title(joint_label, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("大オフセットIC: 真の摩擦トルク vs グレーボックス残差予測(真値軌道上)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/saturation_friction_comparison.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/saturation_friction_comparison.png")

    print(f"\n摩擦推定誤差RMS(全期間): joint1={np.sqrt((residual_error[:, 0]**2).mean()):.4f} "
          f"joint2={np.sqrt((residual_error[:, 1]**2).mean()):.4f}")
    print(f"角速度レンジ: q1_dot=[{q_dot[:, 0].min():.3f}, {q_dot[:, 0].max():.3f}] "
          f"q2_dot=[{q_dot[:, 1].min():.3f}, {q_dot[:, 1].max():.3f}]")


if __name__ == "__main__":
    main()
