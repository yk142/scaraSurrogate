"""Phase3-M1 (#50): ヨー+肩・肘の3D空間マニピュレータの真値シミュレータの検証プロット。

無トルク・無摩擦で腕を振らせ、力学的全エネルギー(運動+位置)の保存と、
ヨー回転を含む3自由度の自由振動を可視化する。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.physics_3d import simulate, total_energy

ZERO_FRICTION = np.zeros(3)
OUT_DIR = "outputs"


def main() -> None:
    initial_state = np.array([0.2, 0.3, -0.5, 0.8, 1.0, -0.7])
    dt = 0.005
    n_steps = 4000  # 20秒分
    t = np.arange(n_steps + 1) * dt

    traj = simulate(initial_state, dt, n_steps, tau=None, c_viscous=ZERO_FRICTION, c_coulomb=ZERO_FRICTION)
    e = total_energy(traj)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(t, traj[:, 0], label="q0(ヨー)")
    axes[0].plot(t, traj[:, 1], label="q1(肩)")
    axes[0].plot(t, traj[:, 2], label="q2(肘)")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("angle [rad]")
    axes[0].set_title("無トルク・無摩擦での自由振動")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, e)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("力学的全エネルギー [J]")
    axes[1].set_title(f"エネルギー保存(相対変動: {np.max(np.abs(e - e[0])) / abs(e[0]):.2e})")

    fig.suptitle("Phase3-M1: ヨー+肩・肘の3D空間マニピュレータの検証")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/yaw_physics_verification.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/yaw_physics_verification.png")


if __name__ == "__main__":
    main()
