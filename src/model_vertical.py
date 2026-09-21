"""Phase2: 重力あり垂直面2関節アーム用のグレーボックスNSSモデル。

Phase 1の`src/model.py`にある慣性行列M(q)・コリオリ行列C(q,q_dot)のtorch実装、
状態エンコーディング、摩擦のsign近似特徴量はそのまま再利用し、重力項G(q)の
torch版を追加する。ブラックボックス(`NSSModel`)はシステムに依存しない汎用MLPの
ため、Phase 1のものをそのまま再利用できる(このファイルでは再実装しない)。
"""
import torch

from src.model import (
    CONTROL_DIM,
    STATE_ENC_DIM,
    AutoregressiveModel,
    _make_mlp,
    coriolis_matrix_torch,
    encode_state,
    friction_sign_features,
    mass_matrix_torch,
)
from src.physics import LC1, LC2, L1, M1, M2
from src.physics_vertical import GRAVITY

DT = 0.02


def gravity_vector_torch(q1: torch.Tensor, q2: torch.Tensor, g: float = GRAVITY) -> torch.Tensor:
    """重力トルクG(q)のtorch版(physics_vertical.gravity_vectorと同じ式)。-> (..., 2)"""
    g1 = (M1 * LC1 + M2 * L1) * g * torch.cos(q1) + M2 * LC2 * g * torch.cos(q1 + q2)
    g2 = M2 * LC2 * g * torch.cos(q1 + q2)
    return torch.stack([g1, g2], dim=-1)


class GrayBoxModelVertical(AutoregressiveModel):
    """グレーボックス版(垂直面): M(q), C(q,q_dot), G(q)をハードコードし、
    NNは摩擦による加速度への影響(未知項)を直接学習する。

        q_ddot = M(q)^-1 @ (tau - C(q,q_dot)@q_dot - G(q)) + residual_accel(state,u)

    Phase2-M7 (#35): 当初はトルクの残差を予測してM(q)^-1で加速度に変換して
    いたが、M(q)の対角成分が小さい(特にjoint2)ためトルク領域の予測誤差が
    M(q)^-1を通じて加速度領域・勾配の両方で増幅され、学習が不安定化していた
    (Phase2-M6参照)。残差を加速度領域で直接予測するよう変更し、この増幅を
    回避する。
    """

    def __init__(self, hidden_dim: int = 64, n_hidden_layers: int = 2, dt: float = DT, g: float = GRAVITY):
        super().__init__()
        self.net = _make_mlp(STATE_ENC_DIM + CONTROL_DIM + 2, CONTROL_DIM, hidden_dim, n_hidden_layers)
        self.dt = dt
        self.g = g

    def residual_accel(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        net_in = torch.cat([encode_state(state), u, friction_sign_features(state)], dim=-1)
        return self.net(net_in)

    def dynamics(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q, q_dot = state[..., :2], state[..., 2:]
        q1, q2 = q[..., 0], q[..., 1]
        q1_dot, q2_dot = q_dot[..., 0], q_dot[..., 1]

        M = mass_matrix_torch(q2)
        C = coriolis_matrix_torch(q2, q1_dot, q2_dot)
        Cq_dot = torch.einsum("...ij,...j->...i", C, q_dot)
        G = gravity_vector_torch(q1, q2, self.g)
        residual = self.residual_accel(state, u)

        rhs = (u - Cq_dot - G).unsqueeze(-1)
        q_ddot_known = torch.linalg.solve(M, rhs).squeeze(-1)
        q_ddot = q_ddot_known + residual
        return torch.cat([q_dot, q_ddot], dim=-1)

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        u = self._default_u(state, u)
        dt = self.dt
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * dt * k1, u)
        k3 = self.dynamics(state + 0.5 * dt * k2, u)
        k4 = self.dynamics(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
