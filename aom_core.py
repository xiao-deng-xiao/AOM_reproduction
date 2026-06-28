import torch


def apply_aom_mitigation(adv_features, anchor_features, alpha=0.5):
    """
    执行 AOM 特征线性插值/外推修正 (One-step linear movement)

    参数:
        adv_features (torch.Tensor): f_source (被攻击的源特征)
        anchor_features (torch.Tensor): f_anchor (高斯噪声平均后的锚点特征)
        alpha (float): 移动步长超参数
    """
    # 论文 Eq. 6: f_movement = (1 - \alpha) * f_source + \alpha * f_anchor
    mitigated_features = (1 - alpha) * adv_features + alpha * anchor_features

    # 修正后必须做 L2 归一化
    mitigated_features = mitigated_features / mitigated_features.norm(dim=-1, keepdim=True)

    return mitigated_features