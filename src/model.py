"""Neural State Space (NSS) サロゲートモデル(2関節アーム)。

姉妹プロジェクト surrogateRolloutVerification のLESSONS.mdで、既知の物理構造
(慣性行列・コリオリ項)をハードコードしNNは残差のみを学習する「グレーボックス化」が
桁違いに効くと分かっているため、最初からブラックボックス版・グレーボックス版の
両方を用意して比較する。

- `NSSModel` (ブラックボックス): 状態遷移をMLPで丸ごと学習する。
- `GrayBoxModel` (グレーボックス): 慣性行列M(q)・コリオリ行列C(q,q̇)をハードコードし、
  RK4積分で状態遷移させる。NNは関節摩擦トルク(未知項)のみを学習する。

状態 x=[theta1, theta2, theta1_dot, theta2_dot] は周期角度を含むため、
(sin theta1, cos theta1, sin theta2, cos theta2, theta1_dot, theta2_dot) にエンコード
してからネットワークに渡す。
"""
import numpy as np
import torch
import torch.nn as nn

from src.physics import I1, I2, L1, LC1, LC2, M1, M2

STATE_DIM = 4  # theta1, theta2, theta1_dot, theta2_dot
STATE_ENC_DIM = 6  # sin1, cos1, sin2, cos2, theta1_dot, theta2_dot
CONTROL_DIM = 2  # tau1, tau2
DT = 0.02


def encode_state(state: torch.Tensor) -> torch.Tensor:
    """(..., 4) [q1,q2,q1_dot,q2_dot] -> (..., 6) [sin1,cos1,sin2,cos2,q1_dot,q2_dot]"""
    q1, q2, q1_dot, q2_dot = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    return torch.stack(
        [torch.sin(q1), torch.cos(q1), torch.sin(q2), torch.cos(q2), q1_dot, q2_dot], dim=-1
    )


def decode_state(enc: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 4) [q1,q2,q1_dot,q2_dot]"""
    sin1, cos1, sin2, cos2 = enc[..., 0], enc[..., 1], enc[..., 2], enc[..., 3]
    q1_dot, q2_dot = enc[..., 4], enc[..., 5]
    q1 = torch.atan2(sin1, cos1)
    q2 = torch.atan2(sin2, cos2)
    return torch.stack([q1, q2, q1_dot, q2_dot], dim=-1)


def _make_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


FRICTION_SIGN_EPS = 0.05  # sign(q_dot)の滑らかな近似 tanh(q_dot/eps) のスケール


def friction_sign_features(state: torch.Tensor) -> torch.Tensor:
    """(..., 4) state -> (..., 2) [tanh(q1_dot/eps), tanh(q2_dot/eps)]

    M5 (#9) で、残差ネットが q_dot≈0 付近のクーロン摩擦 sign(q_dot) の不連続性を
    学習しきれず系統的に過小評価することが判明したため、sign(q_dot)の滑らかな
    近似を明示的な入力特徴量として与える。
    """
    q_dot = state[..., 2:]
    return torch.tanh(q_dot / FRICTION_SIGN_EPS)


def mass_matrix_torch(q2: torch.Tensor) -> torch.Tensor:
    """慣性行列 M(q) のtorch版(physics.mass_matrixと同じ式)。q2: (...,) -> (...,2,2)"""
    cos_q2 = torch.cos(q2)
    m11 = I1 + I2 + M1 * LC1**2 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * cos_q2)
    m12 = I2 + M2 * (LC2**2 + L1 * LC2 * cos_q2)
    m22 = torch.full_like(m11, I2 + M2 * LC2**2)
    row0 = torch.stack([m11, m12], dim=-1)
    row1 = torch.stack([m12, m22], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def coriolis_matrix_torch(q2: torch.Tensor, q1_dot: torch.Tensor, q2_dot: torch.Tensor) -> torch.Tensor:
    """コリオリ・遠心力行列 C(q,q̇) のtorch版。 -> (...,2,2)"""
    h = -M2 * L1 * LC2 * torch.sin(q2)
    c11 = h * q2_dot
    c12 = h * (q1_dot + q2_dot)
    c21 = -h * q1_dot
    c22 = torch.zeros_like(h)
    row0 = torch.stack([c11, c12], dim=-1)
    row1 = torch.stack([c21, c22], dim=-1)
    return torch.stack([row0, row1], dim=-2)


class AutoregressiveModel(nn.Module):
    """1ステップ遷移 `step` を持つモデルに、自己回帰ロールアウトを与える基底クラス。"""

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
        """numpy initial_state (..., 4) から自己回帰的にn_steps展開する(評価用、勾配なし)。

        tau_seq: shape (n_steps, 2) または (n_steps, ..., 2)。Noneならu=0(無入力)。

        Returns: shape (n_steps + 1, ..., 4)
        """
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
        """torch state (..., 4) から勾配を保持したまま自己回帰的にn_steps展開する(学習用)。

        tau_seq: shape (n_steps, ..., 2)。

        Returns: shape (n_steps + 1, ..., 4) (初期状態を含む)
        """
        traj = [state]
        for t in range(n_steps):
            state = self.step(state, tau_seq[t])
            traj.append(state)
        return torch.stack(traj, dim=0)


class NSSModel(AutoregressiveModel):
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


class GrayBoxModel(AutoregressiveModel):
    """グレーボックス版: 慣性行列M(q)・コリオリ行列C(q,q̇)をハードコードし、
    NNは関節摩擦トルク(未知項)のみを学習する。

        M(q) q_ddot + C(q,q_dot) q_dot + residual_torque(state,u) = tau

    真の系は粘性+クーロン摩擦を持つ(physics.py)が、グレーボックスはそれを知らない。
    """

    def __init__(self, hidden_dim: int = 64, n_hidden_layers: int = 2, dt: float = DT):
        super().__init__()
        # 入力: encode_state + u + sign(q_dot)の滑らかな近似(2次元, M6 #11)
        self.net = _make_mlp(STATE_ENC_DIM + CONTROL_DIM + 2, CONTROL_DIM, hidden_dim, n_hidden_layers)
        self.dt = dt

    def residual_torque(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        net_in = torch.cat([encode_state(state), u, friction_sign_features(state)], dim=-1)
        return self.net(net_in)

    def dynamics(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q, q_dot = state[..., :2], state[..., 2:]
        q2 = q[..., 1]
        q1_dot, q2_dot = q_dot[..., 0], q_dot[..., 1]

        M = mass_matrix_torch(q2)
        C = coriolis_matrix_torch(q2, q1_dot, q2_dot)
        Cq_dot = torch.einsum("...ij,...j->...i", C, q_dot)
        residual = self.residual_torque(state, u)

        rhs = (u - Cq_dot - residual).unsqueeze(-1)
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
