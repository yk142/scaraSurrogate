"""Phase2-M14 (#47): 摩擦の構造化(M8, #36)はC_COULOMB比率修正(M13, #45)後も
必要かをアブレーションで検証する。

M13でC_COULOMB/C_VISCOUS比を0.4に回復させたところPTP成功数12/12・整定時間差
0.01秒未満という劇的な改善が得られた。この比率修正があれば、構造化前の
自由なMLP残差(GrayBoxModelVertical)でも十分なのではないか、という疑問を
検証する。

比率修正後の物理パラメータのまま、構造化モデル(StructuredFrictionGrayBoxModelVertical)
と自由なMLP残差(GrayBoxModelVertical)を同じ学習レシピ・同じ評価で比較する。
"""
import numpy as np
import torch

from src.control import angular_error
from src.control_vertical import PTPControllerVertical, run_ptp_surrogate
from src.evaluate_ptp_vertical import N_STEPS, SUCCESS_THRESHOLD, TARGET, WINDOW, make_test_ics
from src.model_vertical import GrayBoxModelVertical, StructuredFrictionGrayBoxModelVertical
from src.train_vertical import CURRICULUM, DT, SEED, train


def evaluate_success_count(model) -> int:
    n_ok = 0
    for ic in make_test_ics():
        traj = run_ptp_surrogate(model, ic, TARGET, N_STEPS, DT, PTPControllerVertical())
        err = np.abs(angular_error(traj[-WINDOW:, :2], TARGET)).sum(axis=-1).mean()
        n_ok += err < SUCCESS_THRESHOLD
    return n_ok


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    free_mlp = train(curriculum=CURRICULUM, model_cls=GrayBoxModelVertical)
    free_mlp_success = evaluate_success_count(free_mlp)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    structured = train(curriculum=CURRICULUM, model_cls=StructuredFrictionGrayBoxModelVertical)
    structured_success = evaluate_success_count(structured)

    print("\n=== アブレーション結果(いずれもC_COULOMB=0.4の比率修正後) ===")
    print(f"GrayBoxModelVertical(自由なMLP): {free_mlp_success}/12")
    print(f"StructuredFrictionGrayBoxModelVertical(構造化): {structured_success}/12")


if __name__ == "__main__":
    main()
