"""Phase2: 垂直面アーム用のNSSモデル学習スクリプト。

学習ループそのものはPhase 1(`src/train.py`)と同じレシピ(マルチステップ損失+
カリキュラム学習)を用い、データセット生成と既定モデルクラスのみ差し替える。
"""
import numpy as np
import torch
import torch.nn as nn

from src.dataset import make_rollout_windows
from src.dataset_vertical import generate_controlled_trajectories
from src.model import AutoregressiveModel, encode_state
from src.model_vertical import GrayBoxModelVertical

DT = 0.02
N_STEPS_PER_TRAJ = 50
# Phase2-M6 (#33): データ量(1200)・エポック数を増やしてみたが、k=30ステージ
# の学習がかえって不安定化しPTP成功数が悪化した(5/12→0/12)。既定値に戻す。
N_TRAIN_TRAJ = 600
N_VAL_TRAJ = 80
SEED = 0
BATCH_SIZE = 512
LR = 1e-3
GRAD_CLIP_NORM = 1.0

K_MAX = 30
WINDOW_STRIDE = 2

CURRICULUM = [(1, 40), (3, 30), (5, 30), (10, 30), (20, 30), (K_MAX, 40)]


def train(
    device: str = "cpu",
    curriculum: list[tuple[int, int]] | None = None,
    model_cls: type[AutoregressiveModel] = GrayBoxModelVertical,
) -> AutoregressiveModel:
    curriculum = curriculum if curriculum is not None else CURRICULUM
    k_max = max(k for k, _ in curriculum)

    train_traj, train_tau = generate_controlled_trajectories(N_TRAIN_TRAJ, DT, N_STEPS_PER_TRAJ, seed=SEED)
    val_traj, val_tau = generate_controlled_trajectories(N_VAL_TRAJ, DT, N_STEPS_PER_TRAJ, seed=SEED + 1)

    x0_train, u_train, targets_train = make_rollout_windows(train_traj, train_tau, k_max, stride=WINDOW_STRIDE)
    x0_val, u_val, targets_val = make_rollout_windows(val_traj, val_tau, k_max, stride=WINDOW_STRIDE)

    x0_train_t = torch.as_tensor(x0_train, dtype=torch.float32, device=device)
    u_train_t = torch.as_tensor(u_train, dtype=torch.float32, device=device)
    target_enc_train = encode_state(torch.as_tensor(targets_train, dtype=torch.float32, device=device))

    x0_val_t = torch.as_tensor(x0_val, dtype=torch.float32, device=device)
    u_val_t = torch.as_tensor(u_val, dtype=torch.float32, device=device)
    target_enc_val = encode_state(torch.as_tensor(targets_val, dtype=torch.float32, device=device))

    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    n_samples = x0_train_t.shape[0]
    global_epoch = 0

    for k, n_epochs in curriculum:
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
            if local_epoch % 10 == 0 or local_epoch == n_epochs - 1:
                with torch.no_grad():
                    val_pred_traj = model.rollout_diff(x0_val_t, u_val_t[:k], k)
                    val_pred_enc = encode_state(val_pred_traj[1:])
                    val_loss = loss_fn(val_pred_enc, target_enc_val[:k]).item()
                print(
                    f"k={k:2d} epoch {global_epoch:4d} "
                    f"train_loss={epoch_loss / n_samples:.6f} val_loss={val_loss:.6f}"
                )

    return model
