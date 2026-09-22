"""Phase2-M12 (#43): クーロン摩擦の識別性を調査する。

M10/M11で、Phase2のクーロン係数(真値0.02)がデータ増強・学習ホライズン
延長のいずれでも真値に近づかなかった(0.15〜0.20に高止まり)。ユーザーから
「Phase1(水平2軸)では成功していたはず」という指摘を受け、両者の摩擦
パラメータを比較すると、Phase2-M2でカオス対策として引き上げたC_VISCOUSに
より、クーロン/粘性比がPhase1の0.4からPhase2の0.02へ20分の1に薄まって
いることが判明した。

この比率がクーロン係数の識別性に決定的であることを検証するため、Phase2の
物理パラメータ(C_VISCOUS=1.0はそのまま)で、C_COULOMBだけをPhase1と
同じ比率(0.4)になるよう引き上げたテストデータで学習し、正しく識別
できるかを確認する。
"""
import numpy as np
import torch
import torch.nn as nn

from src.dataset import make_rollout_windows, sample_initial_states, sample_torque_sequence
from src.model import encode_state
from src.model_vertical import StructuredFrictionGrayBoxModelVertical
from src.physics_vertical import simulate
from src.train_vertical import (
    BATCH_SIZE,
    CURRICULUM,
    DT,
    GRAD_CLIP_NORM,
    K_MAX,
    LR,
    N_STEPS_PER_TRAJ,
    N_TRAIN_TRAJ,
    N_VAL_TRAJ,
    SEED,
    WINDOW_STRIDE,
)

# Phase1と同じ比率(coulomb/viscous = 0.4)になるようテスト用にc_coulombを引き上げる。
TEST_C_VISCOUS = np.array([1.0, 1.0])
TEST_C_COULOMB = np.array([0.4, 0.4])
VELOCITY_LIMIT = 10.0
MAX_RESAMPLE_ATTEMPTS = 20


def _sample_one_trajectory(rng: np.random.Generator):
    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        ic = sample_initial_states(1, rng)[0]
        tau_seq = sample_torque_sequence(N_STEPS_PER_TRAJ, rng, tau_range=(-10.0, 10.0))
        traj = simulate(ic, DT, N_STEPS_PER_TRAJ, tau=tau_seq, c_viscous=TEST_C_VISCOUS, c_coulomb=TEST_C_COULOMB)
        if np.abs(traj[:, 2:]).max() <= VELOCITY_LIMIT:
            return traj, tau_seq
    return traj, tau_seq


def generate_test_trajectories(n_trajectories: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    trajectories, tau_seqs = zip(*[_sample_one_trajectory(rng) for _ in range(n_trajectories)])
    return np.stack(trajectories, axis=0), np.stack(tau_seqs, axis=0)


def main() -> None:
    train_traj, train_tau = generate_test_trajectories(N_TRAIN_TRAJ, SEED)
    val_traj, val_tau = generate_test_trajectories(N_VAL_TRAJ, SEED + 1)

    x0_train, u_train, targets_train = make_rollout_windows(train_traj, train_tau, K_MAX, stride=WINDOW_STRIDE)
    x0_val, u_val, targets_val = make_rollout_windows(val_traj, val_tau, K_MAX, stride=WINDOW_STRIDE)

    x0_train_t = torch.as_tensor(x0_train, dtype=torch.float32)
    u_train_t = torch.as_tensor(u_train, dtype=torch.float32)
    target_enc_train = encode_state(torch.as_tensor(targets_train, dtype=torch.float32))
    x0_val_t = torch.as_tensor(x0_val, dtype=torch.float32)
    u_val_t = torch.as_tensor(u_val, dtype=torch.float32)
    target_enc_val = encode_state(torch.as_tensor(targets_val, dtype=torch.float32))

    channel_std = target_enc_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-3)

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (((pred - target) / channel_std) ** 2).mean()

    torch.manual_seed(SEED)
    model = StructuredFrictionGrayBoxModelVertical()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_samples = x0_train_t.shape[0]
    global_epoch = 0

    for k, n_epochs in CURRICULUM:
        for local_epoch in range(n_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, BATCH_SIZE):
                idx = perm[i : i + BATCH_SIZE]
                pred_traj = model.rollout_diff(x0_train_t[idx], u_train_t[:k, idx], k)
                pred_enc = encode_state(pred_traj[1:])
                loss = loss_fn(pred_enc, target_enc_train[:k, idx, :])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                epoch_loss += loss.item() * idx.shape[0]

            global_epoch += 1
            if local_epoch == n_epochs - 1:
                with torch.no_grad():
                    val_pred_traj = model.rollout_diff(x0_val_t, u_val_t[:k], k)
                    val_pred_enc = encode_state(val_pred_traj[1:])
                    val_loss = loss_fn(val_pred_enc, target_enc_val[:k]).item()
                print(f"k={k:3d} epoch {global_epoch:4d} train_loss={epoch_loss/n_samples:.6f} val_loss={val_loss:.6f}")

    print("\n=== クーロン識別性テスト結果 ===")
    print(f"学習されたc_viscous: {torch.exp(model.log_c_viscous).detach().numpy()}  (真値: {TEST_C_VISCOUS})")
    print(f"学習されたc_coulomb: {torch.exp(model.log_c_coulomb).detach().numpy()}  (真値: {TEST_C_COULOMB})")


if __name__ == "__main__":
    main()
