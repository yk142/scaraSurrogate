"""Phase3: ヨー+肩・肘の3D空間マニピュレータ用の重力補償込み計算トルク法PTP制御。

Phase 2の計算トルク法(`src/control_vertical.py`)をそのまま3自由度に拡張する:
    tau = M(q) @ (-Kp*e - Ki*integral(e) - Kd*q_dot) + C(q,q_dot) @ q_dot + G(q)
"""
import numpy as np

from src.control import angular_error
from src.physics_3d import coriolis_matrix_3d, gravity_vector_3d, mass_matrix_3d, rk4_step

# Phase2最終値(KP=30,KI=3.5,KD=22)を出発点にする。
KP = 30.0
KI = 3.5
KD = 22.0
TAU_MAX = 10.0  # データ生成のTAU_RANGEと揃える
INTEGRAL_CLIP = 4.0


class PTPController3D:
    """重力補償込みの計算トルク法によるPTPコントローラ(積分項つき、3自由度)。"""

    def __init__(self, kp: float = KP, ki: float = KI, kd: float = KD, tau_max: float = TAU_MAX):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tau_max = tau_max
        self.integral = np.zeros(3)

    def reset(self) -> None:
        self.integral = np.zeros(3)

    def compute(self, state: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        q, q_dot = state[:3], state[3:]
        e = angular_error(q, target)
        self.integral = np.clip(self.integral + e * dt, -INTEGRAL_CLIP, INTEGRAL_CLIP)

        q_ddot_desired = -self.kp * e - self.ki * self.integral - self.kd * q_dot

        M = mass_matrix_3d(np.array(q[1]), np.array(q[2]))
        C = coriolis_matrix_3d(np.array(q[1]), np.array(q[2]), np.array(q_dot[0]), np.array(q_dot[1]), np.array(q_dot[2]))
        G = gravity_vector_3d(np.array(q[1]), np.array(q[2]))
        tau = M @ q_ddot_desired + C @ q_dot + G
        return np.clip(tau, -self.tau_max, self.tau_max)


def run_ptp_true(
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    dt: float,
    controller: PTPController3D | None = None,
) -> np.ndarray:
    """真の物理モデルを閉ループでPTP制御する。Returns: traj shape (n_steps+1, 6)"""
    controller = controller or PTPController3D()
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
    controller: PTPController3D | None = None,
) -> np.ndarray:
    """NSSサロゲートモデル(model.rollout互換)を閉ループでPTP制御する。Returns: traj shape (n_steps+1, 6)"""
    controller = controller or PTPController3D()
    controller.reset()
    state = initial_state.copy()
    traj = [state.copy()]
    for _ in range(n_steps):
        tau = controller.compute(state, target, dt)
        state = model.rollout(state, 1, tau_seq=tau[None, :])[-1]
        traj.append(state.copy())
    return np.stack(traj, axis=0)
