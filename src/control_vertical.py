"""Phase2: 重力補償込みの計算トルク法によるPTP(点対点)制御。

Phase 1の計算トルク法(`src/control.py`)に重力補償項G(q)を追加する:
    tau = M(q) @ (-Kp*e - Ki*integral(e) - Kd*q_dot) + C(q,q_dot) @ q_dot + G(q)

M(q), C(q,q_dot), G(q) が真の値と一致していれば、Phase 1と同様に積分項なしの
閉ループ誤差ダイナミクスは e_ddot + Kd*e_dot + Kp*e = 0 という単純な2次系になる
(慣性結合・コリオリ項・重力項が構造的に打ち消される)。関節摩擦は制御則が
知らない未知項のため、積分項(Ki)で定常誤差を打ち消す。
"""
import numpy as np

from src.control import angular_error
from src.physics import coriolis_matrix, mass_matrix
from src.physics_vertical import gravity_vector, rk4_step

# Phase2-M3 (#27): Phase 1のゲイン(KP=6,KD=6)では、C_VISCOUS=2.0という大きな
# 粘性摩擦(重力あり系の暴走対策、Phase2-M2参照)に対して収束が遅すぎ/弱すぎ、
# 30秒以内に0/12ICしか収束しなかった。ゲインを大幅に引き上げて解消。
KP = 30.0
KI = 3.5
KD = 22.0
TAU_MAX = 10.0  # データ生成のTAU_RANGEと揃える(重力補償に必要な最大トルクを上回る)
INTEGRAL_CLIP = 4.0


class PTPControllerVertical:
    """重力補償込みの計算トルク法によるPTPコントローラ(積分項つき)。"""

    def __init__(self, kp: float = KP, ki: float = KI, kd: float = KD, tau_max: float = TAU_MAX):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tau_max = tau_max
        self.integral = np.zeros(2)

    def reset(self) -> None:
        self.integral = np.zeros(2)

    def compute(self, state: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        q, q_dot = state[:2], state[2:]
        e = angular_error(q, target)
        self.integral = np.clip(self.integral + e * dt, -INTEGRAL_CLIP, INTEGRAL_CLIP)

        q_ddot_desired = -self.kp * e - self.ki * self.integral - self.kd * q_dot

        M = mass_matrix(np.array(q[1]))
        C = coriolis_matrix(np.array(q[1]), np.array(q_dot[0]), np.array(q_dot[1]))
        G = gravity_vector(np.array(q[0]), np.array(q[1]))
        tau = M @ q_ddot_desired + C @ q_dot + G
        return np.clip(tau, -self.tau_max, self.tau_max)


def run_ptp_true(
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    dt: float,
    controller: PTPControllerVertical | None = None,
) -> np.ndarray:
    """真の物理モデル(垂直面、重力あり)を閉ループでPTP制御する。

    Returns: traj shape (n_steps+1, 4)
    """
    controller = controller or PTPControllerVertical()
    controller.reset()
    state = initial_state.copy()
    traj = [state.copy()]
    for _ in range(n_steps):
        tau = controller.compute(state, target, dt)
        state = rk4_step(state, dt, tau)
        traj.append(state.copy())
    return np.stack(traj, axis=0)


def run_ptp_surrogate(
    model,
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    dt: float,
    controller: PTPControllerVertical | None = None,
) -> np.ndarray:
    """NSSサロゲートモデル(model.rollout互換)を閉ループでPTP制御する。

    Returns: traj shape (n_steps+1, 4)
    """
    controller = controller or PTPControllerVertical()
    controller.reset()
    state = initial_state.copy()
    traj = [state.copy()]
    for _ in range(n_steps):
        tau = controller.compute(state, target, dt)
        state = model.rollout(state, 1, tau_seq=tau[None, :])[-1]
        traj.append(state.copy())
    return np.stack(traj, axis=0)
