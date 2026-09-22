"""Phase3: ヨー+肩・肘の3自由度空間(3D)マニピュレータの真値シミュレータ。

実機ロボットアームの基本構造(ヨー・ショルダー・エルボー)に相当する:
    q0: 基部ヨー関節(鉛直Z軸回り、水平面内を旋回。Phase1の関節と同じ回転方向)
    q1: 肩関節(水平軸回り、q0で回転する鉛直面内で動く。Phase2のq1相当、重力あり)
    q2: 肘関節(q1に対する相対角、同じ鉛直面内。Phase2のq2相当、重力あり)

状態は6次元 x=[q0,q1,q2,q0_dot,q1_dot,q2_dot]、入力は3次元 u=[tau0,tau1,tau2]。

## 力学の導出

リンクを細い一様棒(軸方向の慣性モーメント≈0、軸に垂直な方向はI=m*l^2/12で
対称)と仮定してラグランジュ力学で導出すると、以下がPhase1/Phase2の量を
再利用する形で書ける:

- 質量行列M(q)はヨー(q0)とピッチ(q1,q2)についてブロック対角:
    M = [[M00(q1,q2),      0      ],
         [    0,      M_pitch(q2) ]]
  M_pitch(q2)はPhase1/2の`mass_matrix(q2)`と完全に同一(q0=一定なら平面
  2リンクアームの運動方程式に一致するはずなので、当然の帰結)。

  M00(q1,q2) = A*cos^2(q1) + m2*B(q1,q2)^2 + I2*cos^2(q1+q2)
    A = m1*lc1^2 + I1  (Phase1/2のM11の一部と同じ組み合わせ)
    B(q1,q2) = l1*cos(q1) + lc2*cos(q1+q2)  (腕の水平方向への張り出し半径)

- 重力項G(q)はPhase2の`gravity_vector(q1,q2)`と完全に同一で、q0成分は0
  (ヨー軸は鉛直なので重力トルクを受けない)。

- コリオリ行列C(q,q_dot)は、ピッチ側の2x2ブロックはPhase1/2と同一のh項
  (h=-m2*l1*lc2*sin(q2))を使うが、ヨー速度q0_dotとピッチ速度の間に
  M00のq1,q2依存から生じる遠心力的な結合項が追加される:
    C[0,1] = dM00_dq1 * q0_dot,  C[0,2] = dM00_dq2 * q0_dot
    C[1,0] = -0.5 * dM00_dq1 * q0_dot,  C[2,0] = -0.5 * dM00_dq2 * q0_dot

無トルク・無摩擦なら力学的全エネルギー(運動+位置エネルギー)が保存される
ことが、この導出の正しさの検証になる。
"""
import numpy as np

from src.physics import I1, I2, LC1, LC2, L1, M1, M2, coriolis_matrix, mass_matrix
from src.physics_vertical import GRAVITY, V_STRIBECK, gravity_vector

C_VISCOUS = np.array([1.0, 1.0, 1.0])
C_COULOMB = np.array([0.4, 0.4, 0.4])

_A = M1 * LC1**2 + I1

# Phase3-M2 (#52): 腕がヨー軸(鉛直)と一直線に伸びる姿勢(q1≈-pi/2かつq2≈-pi等)
# では、腕の水平方向への張り出しが0になりM00(ヨー軸まわりの慣性)が0に近づく
# 運動学的特異点が存在する(実機ロボットの手首特異点と同種の現象)。この近傍で
# 質量行列を逆行列計算すると発散する。
# 当初max(m00,floor)というハードクランプで正則化したが、これはラグランジュ
# 力学の構造を破り、特異点通過時に力学的エネルギーが数倍に跳ね上がる
# アーティファクトを生んだ(不連続な変更のため)。代わりに、ヨー軸に常に存在する
# 微小なロータ慣性(モータ・エンコーダ由来、実機でも一般的)としてM00に定数を
# 加算する滑らかな正則化にする。これは有効ラグランジアンへの正当な項の追加で
# あり、通常の動作範囲への影響は無視できるほど小さい。
M00_EPSILON = 0.02


def _arm_radius(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """腕の水平方向への張り出し半径 B(q1,q2) = l1*cos(q1) + lc2*cos(q1+q2)"""
    return L1 * np.cos(q1) + LC2 * np.cos(q1 + q2)


def mass_matrix_3d(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """質量行列M(q)。shape (...,) -> shape (..., 3, 3)。ヨー・ピッチでブロック対角。"""
    B = _arm_radius(q1, q2)
    m00 = _A * np.cos(q1) ** 2 + M2 * B**2 + I2 * np.cos(q1 + q2) ** 2 + M00_EPSILON
    m_pitch = mass_matrix(q2)  # shape (..., 2, 2)

    M = np.zeros(q1.shape + (3, 3))
    M[..., 0, 0] = m00
    M[..., 1:, 1:] = m_pitch
    return M


def _dM00(q1: np.ndarray, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M00のq1, q2に関する偏微分。"""
    B = _arm_radius(q1, q2)
    dB_dq1 = -L1 * np.sin(q1) - LC2 * np.sin(q1 + q2)
    dB_dq2 = -LC2 * np.sin(q1 + q2)

    dM00_dq1 = -_A * np.sin(2 * q1) + 2 * M2 * B * dB_dq1 - I2 * np.sin(2 * (q1 + q2))
    dM00_dq2 = 2 * M2 * B * dB_dq2 - I2 * np.sin(2 * (q1 + q2))
    return dM00_dq1, dM00_dq2


def coriolis_matrix_3d(
    q1: np.ndarray, q2: np.ndarray, q0_dot: np.ndarray, q1_dot: np.ndarray, q2_dot: np.ndarray
) -> np.ndarray:
    """コリオリ・遠心力行列C(q,q_dot)。shape (...,) -> shape (..., 3, 3)。"""
    dM00_dq1, dM00_dq2 = _dM00(q1, q2)
    c_pitch = coriolis_matrix(q2, q1_dot, q2_dot)  # shape (..., 2, 2)

    C = np.zeros(q1.shape + (3, 3))
    C[..., 0, 1] = dM00_dq1 * q0_dot
    C[..., 0, 2] = dM00_dq2 * q0_dot
    C[..., 1, 0] = -0.5 * dM00_dq1 * q0_dot
    C[..., 2, 0] = -0.5 * dM00_dq2 * q0_dot
    C[..., 1:, 1:] = c_pitch
    return C


def gravity_vector_3d(q1: np.ndarray, q2: np.ndarray, g: float = GRAVITY) -> np.ndarray:
    """重力トルクG(q)。q0成分は常に0(ヨー軸は鉛直)。shape (...,) -> shape (..., 3)"""
    g_pitch = gravity_vector(q1, q2, g)  # shape (..., 2)
    zeros = np.zeros(q1.shape + (1,))
    return np.concatenate([zeros, g_pitch], axis=-1)


def friction_torque(
    q_dot: np.ndarray, c_viscous: np.ndarray = C_VISCOUS, c_coulomb: np.ndarray = C_COULOMB
) -> np.ndarray:
    """関節摩擦トルク(粘性+クーロン、Stribeck風の滑らかな近似)。q_dot: (...,3) -> (...,3)"""
    return c_viscous * q_dot + c_coulomb * np.tanh(q_dot / V_STRIBECK)


def dynamics(
    state: np.ndarray,
    tau: np.ndarray,
    c_viscous: np.ndarray = C_VISCOUS,
    c_coulomb: np.ndarray = C_COULOMB,
    g: float = GRAVITY,
) -> np.ndarray:
    """状態の時間微分dx/dtを返す。

    state: shape (..., 6) = [q0,q1,q2,q0_dot,q1_dot,q2_dot]
    tau: shape (..., 3) = [tau0,tau1,tau2]
    """
    q = state[..., :3]
    q_dot = state[..., 3:]
    q1, q2 = q[..., 1], q[..., 2]
    q0_dot, q1_dot, q2_dot = q_dot[..., 0], q_dot[..., 1], q_dot[..., 2]

    M = mass_matrix_3d(q1, q2)
    C = coriolis_matrix_3d(q1, q2, q0_dot, q1_dot, q2_dot)
    G = gravity_vector_3d(q1, q2, g)

    rhs = tau - np.einsum("...ij,...j->...i", C, q_dot) - G - friction_torque(q_dot, c_viscous, c_coulomb)
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
    """RK4で1ステップ積分する。tauはこのステップ内で一定とみなす。"""
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
    """初期状態からn_steps+1点の軌道を生成する。

    tau: shape (3,)(全ステップ一定)、shape (n_steps,3)(区分定数トルク列)、
    またはNone(無トルク)。

    Returns: shape (n_steps+1, 6)
    """
    if tau is None:
        tau_seq = np.zeros((n_steps, 3))
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


def kinetic_energy(state: np.ndarray) -> np.ndarray:
    """全運動エネルギー 0.5 * q_dot^T M(q) q_dot。state: shape (...,6) -> shape (...,)"""
    q1, q2 = state[..., 1], state[..., 2]
    q_dot = state[..., 3:]
    M = mass_matrix_3d(q1, q2)
    Mq = np.einsum("...ij,...j->...i", M, q_dot)
    return 0.5 * np.einsum("...i,...i->...", q_dot, Mq)


def potential_energy(state: np.ndarray, g: float = GRAVITY) -> np.ndarray:
    """位置エネルギー。state: shape (...,6) -> shape (...,)"""
    q1, q2 = state[..., 1], state[..., 2]
    return (M1 * LC1 + M2 * L1) * g * np.sin(q1) + M2 * LC2 * g * np.sin(q1 + q2)


def total_energy(state: np.ndarray, g: float = GRAVITY) -> np.ndarray:
    """力学的全エネルギー(運動エネルギー+位置エネルギー)。"""
    return kinetic_energy(state) + potential_energy(state, g)
