"""水平2関節アーム(平面2リンクマニピュレータ、重力なし)の真値シミュレータ。

状態は x = [theta1, theta2, theta1_dot, theta2_dot]、入力は u = [tau1, tau2]。
運動方程式:
    M(q) q_ddot + C(q, q_dot) q_dot + F(q_dot) = tau

水平面内の運動のため重力項はない。F(q_dot) は粘性摩擦のみ(デフォルト0)。
リンクは一様棒として lc = l/2 (重心は中点), I = m*l^2/12 (重心まわりの慣性モーメント)
をデフォルトパラメータとする。
"""
import numpy as np

# --- リンクパラメータ(一様棒) ---
M1 = 1.0  # リンク1の質量 [kg]
M2 = 1.0  # リンク2の質量 [kg]
L1 = 0.5  # リンク1の長さ [m]
L2 = 0.5  # リンク2の長さ [m]
LC1 = L1 / 2  # リンク1の重心位置(関節1からの距離)
LC2 = L2 / 2  # リンク2の重心位置(関節2からの距離)
I1 = M1 * L1**2 / 12  # リンク1の重心まわり慣性モーメント
I2 = M2 * L2**2 / 12  # リンク2の重心まわり慣性モーメント

C_VISCOUS = 0.0  # 粘性摩擦係数(デフォルトは無摩擦、M1では0)


def mass_matrix(q2: np.ndarray) -> np.ndarray:
    """慣性行列 M(q)。q2 の関数のみ(q1には依存しない)。

    q2: shape (...,) -> shape (..., 2, 2)
    """
    cos_q2 = np.cos(q2)
    m11 = I1 + I2 + M1 * LC1**2 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * cos_q2)
    m12 = I2 + M2 * (LC2**2 + L1 * LC2 * cos_q2)
    m22 = np.broadcast_to(I2 + M2 * LC2**2, m11.shape)
    row0 = np.stack([m11, m12], axis=-1)
    row1 = np.stack([m12, m22], axis=-1)
    return np.stack([row0, row1], axis=-2)


def coriolis_matrix(q2: np.ndarray, q1_dot: np.ndarray, q2_dot: np.ndarray) -> np.ndarray:
    """コリオリ・遠心力行列 C(q, q_dot)。

    q2, q1_dot, q2_dot: shape (...,) -> shape (..., 2, 2)
    """
    h = -M2 * L1 * LC2 * np.sin(q2)
    c11 = h * q2_dot
    c12 = h * (q1_dot + q2_dot)
    c21 = -h * q1_dot
    c22 = np.zeros_like(h)
    row0 = np.stack([c11, c12], axis=-1)
    row1 = np.stack([c21, c22], axis=-1)
    return np.stack([row0, row1], axis=-2)


def dynamics(state: np.ndarray, tau: np.ndarray, c: float = C_VISCOUS) -> np.ndarray:
    """状態の時間微分 dx/dt を返す。

    state: shape (..., 4) = [theta1, theta2, theta1_dot, theta2_dot]
    tau: shape (..., 2) = [tau1, tau2]
    """
    q = state[..., :2]
    q_dot = state[..., 2:]
    q2 = q[..., 1]
    q1_dot, q2_dot = q_dot[..., 0], q_dot[..., 1]

    M = mass_matrix(q2)
    C = coriolis_matrix(q2, q1_dot, q2_dot)

    friction = c * q_dot
    rhs = tau - np.einsum("...ij,...j->...i", C, q_dot) - friction
    q_ddot = np.linalg.solve(M, rhs)

    return np.concatenate([q_dot, q_ddot], axis=-1)


def rk4_step(state: np.ndarray, dt: float, tau: np.ndarray, c: float = C_VISCOUS) -> np.ndarray:
    """RK4で1ステップ積分する。tau はこのステップ内で一定とみなす。"""
    k1 = dynamics(state, tau, c)
    k2 = dynamics(state + 0.5 * dt * k1, tau, c)
    k3 = dynamics(state + 0.5 * dt * k2, tau, c)
    k4 = dynamics(state + dt * k3, tau, c)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    initial_state: np.ndarray,
    dt: float,
    n_steps: int,
    tau: np.ndarray | None = None,
    c: float = C_VISCOUS,
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
        state = rk4_step(state, dt, tau_seq[t], c)
        traj.append(state)
    return np.stack(traj, axis=0)


def kinetic_energy(state: np.ndarray) -> np.ndarray:
    """全運動エネルギー 0.5 * q_dot^T M(q) q_dot。水平面のため位置エネルギーはない。

    state: shape (..., 4) -> shape (...,)
    """
    q2 = state[..., 1]
    q_dot = state[..., 2:]
    M = mass_matrix(q2)
    Mq = np.einsum("...ij,...j->...i", M, q_dot)
    return 0.5 * np.einsum("...i,...i->...", q_dot, Mq)
