"""計算トルク法(computed torque control)によるPTP(点対点)制御。

既知の慣性行列M(q)・コリオリ行列C(q,q_dot)を使って関節空間を線形化する:
    tau = M(q) @ (-Kp*e - Ki*integral(e) - Kd*q_dot) + C(q,q_dot) @ q_dot
ここで e = q - target(周期角度なのでwrap対応)。

M(q), C(q,q_dot) が真の値と一致していれば、積分項なしの閉ループ誤差ダイナミクスは
e_ddot + Kd*e_dot + Kp*e = 0 という単純な2次系になる(関節間の慣性結合・コリオリ項が
構造的に打ち消される)。ただし関節摩擦(クーロン摩擦)は制御則が知らない未知項のため、
比例微分項だけでは静止時に定常誤差(スティクション由来)が残る。これを打ち消すために
積分項(Ki)を加える。グレーボックスサロゲートと全く同じ既知構造(M(q), C(q,q_dot))を
制御則にも使うという一貫した設計になっている。
"""
import numpy as np

from src.physics import coriolis_matrix, mass_matrix, rk4_step

KP = 6.0
KI = 0.3
KD = 6.0
TAU_MAX = 1.0  # 学習データのTAU_RANGEと揃える(この範囲外は未学習領域になる)
INTEGRAL_CLIP = 2.0


def angular_error(q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """q と target の符号付き誤差(各関節、wrap済みで(-pi,pi]に収まる)。"""
    return np.arctan2(np.sin(q - target), np.cos(q - target))


class PTPController:
    """計算トルク法によるPTPコントローラ(積分項つき)。"""

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
        tau = M @ q_ddot_desired + C @ q_dot
        return np.clip(tau, -self.tau_max, self.tau_max)


def run_ptp_true(
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    dt: float,
    controller: PTPController | None = None,
) -> np.ndarray:
    """真の物理モデルを閉ループでPTP制御する。

    Returns: traj shape (n_steps+1, 4)
    """
    controller = controller or PTPController()
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
    controller: PTPController | None = None,
) -> np.ndarray:
    """NSSサロゲートモデル(model.rollout互換)を閉ループでPTP制御する。

    Returns: traj shape (n_steps+1, 4)
    """
    controller = controller or PTPController()
    controller.reset()
    state = initial_state.copy()
    traj = [state.copy()]
    for _ in range(n_steps):
        tau = controller.compute(state, target, dt)
        state = model.rollout(state, 1, tau_seq=tau[None, :])[-1]
        traj.append(state.copy())
    return np.stack(traj, axis=0)
