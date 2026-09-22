"""Phase3: ヨー+肩・肘の3D空間マニピュレータ用のNSSモデル。

Phase 1/2の`src/model.py`にある慣性行列M(q)・コリオリ行列C(q,q_dot)のtorch
実装(ピッチ側)、`src/model_vertical.py`の重力項G(q)のtorch実装をそのまま
再利用し、ヨー(q0)とピッチ(q1,q2)の結合項のみ新規実装する。

状態は6次元 x=[q0,q1,q2,q0_dot,q1_dot,q2_dot]。周期角度(q0,q1,q2)を含むため、
(sin q0, cos q0, sin q1, cos q1, sin q2, cos q2, q0_dot, q1_dot, q2_dot) の
9次元にエンコードしてからネットワークに渡す。入力は3次元 u=[tau0,tau1,tau2]。

Phase2-M8/M12/M13/M14の教訓を最初から適用し、摩擦は自由なMLPではなく
構造化モデル(関数形固定、係数のみ学習)のみを用意する。
"""
import numpy as np
import torch
import torch.nn as nn

from src.model import _make_mlp, coriolis_matrix_torch, mass_matrix_torch
from src.model_vertical import gravity_vector_torch
from src.physics import I1, I2, LC1, LC2, L1, M1, M2
from src.physics_3d import M00_EPSILON
from src.physics_vertical import GRAVITY, V_STRIBECK

STATE_DIM = 6  # q0, q1, q2, q0_dot, q1_dot, q2_dot
STATE_ENC_DIM = 9  # sin0,cos0,sin1,cos1,sin2,cos2, q0_dot,q1_dot,q2_dot
CONTROL_DIM = 3  # tau0, tau1, tau2
DT = 0.02

_A = M1 * LC1**2 + I1


def encode_state(state: torch.Tensor) -> torch.Tensor:
    """(..., 6) [q0,q1,q2,q0d,q1d,q2d] -> (..., 9)"""
    q0, q1, q2 = state[..., 0], state[..., 1], state[..., 2]
    q0_dot, q1_dot, q2_dot = state[..., 3], state[..., 4], state[..., 5]
    return torch.stack(
        [
            torch.sin(q0), torch.cos(q0),
            torch.sin(q1), torch.cos(q1),
            torch.sin(q2), torch.cos(q2),
            q0_dot, q1_dot, q2_dot,
        ],
        dim=-1,
    )


def decode_state(enc: torch.Tensor) -> torch.Tensor:
    """(..., 9) -> (..., 6) [q0,q1,q2,q0d,q1d,q2d]"""
    sin0, cos0, sin1, cos1, sin2, cos2 = (enc[..., i] for i in range(6))
    q0_dot, q1_dot, q2_dot = enc[..., 6], enc[..., 7], enc[..., 8]
    q0 = torch.atan2(sin0, cos0)
    q1 = torch.atan2(sin1, cos1)
    q2 = torch.atan2(sin2, cos2)
    return torch.stack([q0, q1, q2, q0_dot, q1_dot, q2_dot], dim=-1)


def _arm_radius_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return L1 * torch.cos(q1) + LC2 * torch.cos(q1 + q2)


def mass_matrix_3d_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """質量行列M(q)のtorch版(physics_3d.mass_matrix_3dと同じ式)。-> (...,3,3)"""
    B = _arm_radius_torch(q1, q2)
    m00 = _A * torch.cos(q1) ** 2 + M2 * B**2 + I2 * torch.cos(q1 + q2) ** 2 + M00_EPSILON
    m_pitch = mass_matrix_torch(q2)  # (...,2,2)

    M = torch.zeros(q1.shape + (3, 3), dtype=q1.dtype, device=q1.device)
    M[..., 0, 0] = m00
    M[..., 1:, 1:] = m_pitch
    return M


def _dM00_torch(q1: torch.Tensor, q2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B = _arm_radius_torch(q1, q2)
    dB_dq1 = -L1 * torch.sin(q1) - LC2 * torch.sin(q1 + q2)
    dB_dq2 = -LC2 * torch.sin(q1 + q2)

    dM00_dq1 = -_A * torch.sin(2 * q1) + 2 * M2 * B * dB_dq1 - I2 * torch.sin(2 * (q1 + q2))
    dM00_dq2 = 2 * M2 * B * dB_dq2 - I2 * torch.sin(2 * (q1 + q2))
    return dM00_dq1, dM00_dq2


def coriolis_matrix_3d_torch(
    q1: torch.Tensor, q2: torch.Tensor, q0_dot: torch.Tensor, q1_dot: torch.Tensor, q2_dot: torch.Tensor
) -> torch.Tensor:
    """コリオリ・遠心力行列C(q,q_dot)のtorch版(physics_3d.coriolis_matrix_3dと同じ式)。-> (...,3,3)"""
    dM00_dq1, dM00_dq2 = _dM00_torch(q1, q2)
    c_pitch = coriolis_matrix_torch(q2, q1_dot, q2_dot)  # (...,2,2)

    C = torch.zeros(q1.shape + (3, 3), dtype=q1.dtype, device=q1.device)
    C[..., 0, 1] = dM00_dq1 * q0_dot
    C[..., 0, 2] = dM00_dq2 * q0_dot
    C[..., 1, 0] = -0.5 * dM00_dq1 * q0_dot
    C[..., 2, 0] = -0.5 * dM00_dq2 * q0_dot
    C[..., 1:, 1:] = c_pitch
    return C


def gravity_vector_3d_torch(q1: torch.Tensor, q2: torch.Tensor, g: float = GRAVITY) -> torch.Tensor:
    """重力トルクG(q)のtorch版。q0成分は常に0。-> (...,3)"""
    g_pitch = gravity_vector_torch(q1, q2, g)  # (...,2)
    zeros = torch.zeros(q1.shape + (1,), dtype=q1.dtype, device=q1.device)
    return torch.cat([zeros, g_pitch], dim=-1)


class AutoregressiveModel3D(nn.Module):
    """3D空間マニピュレータ用の自己回帰ロールアウト基底クラス(CONTROL_DIM=3)。"""

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _default_u(state: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
        if u is not None:
            return u
        return torch.zeros(*state.shape[:-1], CONTROL_DIM, device=state.device)

    @torch.no_grad()
    def rollout(
        self, initial_state: np.ndarray, n_steps: int, tau_seq: np.ndarray | None = None
    ) -> np.ndarray:
        device = next(self.parameters()).device
        state = torch.as_tensor(initial_state, dtype=torch.float32, device=device)
        traj = [state]
        for t in range(n_steps):
            u = None if tau_seq is None else torch.as_tensor(
                tau_seq[t], dtype=torch.float32, device=device
            ).expand(*state.shape[:-1], CONTROL_DIM)
            state = self.step(state, u)
            traj.append(state)
        return torch.stack(traj, dim=0).cpu().numpy()

    def rollout_diff(self, state: torch.Tensor, tau_seq: torch.Tensor, n_steps: int) -> torch.Tensor:
        traj = [state]
        for t in range(n_steps):
            state = self.step(state, tau_seq[t])
            traj.append(state)
        return torch.stack(traj, dim=0)


class NSSModel3D(AutoregressiveModel3D):
    """ブラックボックス版: 状態遷移そのものをMLPで学習する。"""

    def __init__(self, hidden_dim: int = 64, n_hidden_layers: int = 2):
        super().__init__()
        self.net = _make_mlp(STATE_ENC_DIM + CONTROL_DIM, STATE_ENC_DIM, hidden_dim, n_hidden_layers)

    def forward(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        enc = encode_state(state)
        net_in = torch.cat([enc, self._default_u(state, u)], dim=-1)
        return self.net(net_in)

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        return decode_state(self.forward(state, u))


class StructuredFrictionGrayBoxModel3D(AutoregressiveModel3D):
    """グレーボックス版(3D、摩擦を構造化): M(q), C(q,q_dot), G(q)に加え、摩擦の
    関数形(粘性+クーロン、Stribeck風の滑らかな近似)自体もハードコードし、
    NNは使わず係数(関節ごと1個、計6個)だけを学習パラメータにする
    (Phase2-M8/M12/M13/M14の教訓を最初から適用)。
    """

    def __init__(self, dt: float = DT, g: float = GRAVITY, v_stribeck: float = V_STRIBECK):
        super().__init__()
        self.log_c_viscous = nn.Parameter(torch.zeros(3))
        self.log_c_coulomb = nn.Parameter(torch.zeros(3))
        self.dt = dt
        self.g = g
        self.v_stribeck = v_stribeck

    def residual_torque(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q_dot = state[..., 3:]
        c_viscous = torch.exp(self.log_c_viscous)
        c_coulomb = torch.exp(self.log_c_coulomb)
        return c_viscous * q_dot + c_coulomb * torch.tanh(q_dot / self.v_stribeck)

    def dynamics(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q, q_dot = state[..., :3], state[..., 3:]
        q1, q2 = q[..., 1], q[..., 2]
        q0_dot, q1_dot, q2_dot = q_dot[..., 0], q_dot[..., 1], q_dot[..., 2]

        M = mass_matrix_3d_torch(q1, q2)
        C = coriolis_matrix_3d_torch(q1, q2, q0_dot, q1_dot, q2_dot)
        Cq_dot = torch.einsum("...ij,...j->...i", C, q_dot)
        G = gravity_vector_3d_torch(q1, q2, self.g)
        residual = self.residual_torque(state, u)

        rhs = (u - Cq_dot - G - residual).unsqueeze(-1)
        q_ddot = torch.linalg.solve(M, rhs).squeeze(-1)
        return torch.cat([q_dot, q_ddot], dim=-1)

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        u = self._default_u(state, u)
        dt = self.dt
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * dt * k1, u)
        k3 = self.dynamics(state + 0.5 * dt * k2, u)
        k4 = self.dynamics(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
