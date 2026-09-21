"""重力あり垂直面2関節アーム(Phase 2)の真値シミュレータ。

水平面版(`src/physics.py`, Phase 1)との違いは重力項G(q)のみ。慣性行列M(q)・
コリオリ行列C(q,q_dot)は角度の取り方(q1: 水平から測った絶対角、q2: リンク1に
対する相対角)が同じであれば同一の式になるため、`src/physics.py`のものを
そのまま再利用する。

運動方程式:
    M(q) q_ddot + C(q, q_dot) q_dot + G(q) + F(q_dot) = tau

G(q)は標準的な2リンクマニピュレータの重力項(一様棒、重心lc=l/2):
    G1 = (m1*lc1 + m2*l1) * g * cos(q1) + m2*lc2*g*cos(q1+q2)
    G2 = m2*lc2*g*cos(q1+q2)
"""
import numpy as np

from src.physics import LC1, LC2, L1, M1, M2, coriolis_matrix, kinetic_energy, mass_matrix

GRAVITY = 9.81

# Phase2-M2 (#25): 水平面版と同じ摩擦係数(0.05)では、重力+リンク2の小さい慣性に
# よるカオス的な二重振り子効果でランダムIC・ランダムトルクの軌道がほぼ全て
# 角速度数十rad/s(非現実的)まで発散することが判明した。実機ロボットアームの
# 減速機・モータ由来の粘性摩擦を模して大幅に引き上げ、現実的な速度域に収める。
C_VISCOUS = np.array([2.0, 2.0])
C_COULOMB = np.array([0.02, 0.02])
V_STRIBECK = 0.03


def gravity_vector(q1: np.ndarray, q2: np.ndarray, g: float = GRAVITY) -> np.ndarray:
    """重力トルクG(q)。q1, q2: shape (...,) -> shape (..., 2)"""
    g1 = (M1 * LC1 + M2 * L1) * g * np.cos(q1) + M2 * LC2 * g * np.cos(q1 + q2)
    g2 = M2 * LC2 * g * np.cos(q1 + q2)
    return np.stack([g1, g2], axis=-1)


def potential_energy(state: np.ndarray, g: float = GRAVITY) -> np.ndarray:
    """位置エネルギー(水平基準面からの高さ×重力)。state: shape (..., 4) -> shape (...,)"""
    q1, q2 = state[..., 0], state[..., 1]
    return (M1 * LC1 + M2 * L1) * g * np.sin(q1) + M2 * LC2 * g * np.sin(q1 + q2)


def total_energy(state: np.ndarray, g: float = GRAVITY) -> np.ndarray:
    """力学的全エネルギー(運動エネルギー+位置エネルギー)。"""
    return kinetic_energy(state) + potential_energy(state, g)


def friction_torque(
    q_dot: np.ndarray, c_viscous: np.ndarray = C_VISCOUS, c_coulomb: np.ndarray = C_COULOMB
) -> np.ndarray:
    """関節摩擦トルク(粘性+クーロン、Stribeck風の滑らかな近似)。水平面版M11と同じ式。"""
    return c_viscous * q_dot + c_coulomb * np.tanh(q_dot / V_STRIBECK)


def dynamics(
    state: np.ndarray,
    tau: np.ndarray,
    c_viscous: np.ndarray = C_VISCOUS,
    c_coulomb: np.ndarray = C_COULOMB,
    g: float = GRAVITY,
) -> np.ndarray:
    """状態の時間微分 dx/dt を返す。

    state: shape (..., 4) = [theta1, theta2, theta1_dot, theta2_dot]
    tau: shape (..., 2) = [tau1, tau2]
    """
    q = state[..., :2]
    q_dot = state[..., 2:]
    q1, q2 = q[..., 0], q[..., 1]
    q1_dot, q2_dot = q_dot[..., 0], q_dot[..., 1]

    M = mass_matrix(q2)
    C = coriolis_matrix(q2, q1_dot, q2_dot)
    G = gravity_vector(q1, q2, g)

    rhs = (
        tau
        - np.einsum("...ij,...j->...i", C, q_dot)
        - G
        - friction_torque(q_dot, c_viscous, c_coulomb)
    )
    q_ddot = np.linalg.solve(M, rhs)

    return np.concatenate([q_dot, q_ddot], axis=-1)


def rk4_step(
    state: np.ndarray,
    dt: float,
    tau: np.ndarray,
    c_viscous: np.ndarray = C_VISCOUS,
    c_coulomb: np.ndarray = C_COULOMB,
    g: float = GRAVITY,
) -> np.ndarray:
    """RK4で1ステップ積分する。tau はこのステップ内で一定とみなす。"""
    k1 = dynamics(state, tau, c_viscous, c_coulomb, g)
    k2 = dynamics(state + 0.5 * dt * k1, tau, c_viscous, c_coulomb, g)
    k3 = dynamics(state + 0.5 * dt * k2, tau, c_viscous, c_coulomb, g)
    k4 = dynamics(state + dt * k3, tau, c_viscous, c_coulomb, g)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    initial_state: np.ndarray,
    dt: float,
    n_steps: int,
    tau: np.ndarray | None = None,
    c_viscous: np.ndarray = C_VISCOUS,
    c_coulomb: np.ndarray = C_COULOMB,
    g: float = GRAVITY,
) -> np.ndarray:
    """初期状態から n_steps + 1 点の軌道を生成する。

    tau: shape (2,)(全ステップ一定)、shape (n_steps, 2)(区分定数トルク列)、
    または None(無トルク)。

    Returns: shape (n_steps + 1, 4)
    """
    if tau is None:
        tau_seq = np.zeros((n_steps, 2))
    elif tau.ndim == 1:
        tau_seq = np.tile(tau, (n_steps, 1))
    else:
        tau_seq = tau

    traj = [initial_state]
    state = initial_state
    for t in range(n_steps):
        state = rk4_step(state, dt, tau_seq[t], c_viscous, c_coulomb, g)
        traj.append(state)
    return np.stack(traj, axis=0)
