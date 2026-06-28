import argparse
import time
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Subset
import torchattacks
import clip

# 导入你 R-TPT 库里的工具
from data.imagnet_prompts import imagenet_classes
from data.datautils import build_dataset
from utils.tools import AverageMeter, ProgressMeter, accuracy, set_random_seed
from data.cls_to_names import *

# 导入你自己的 AOM 防御算法
from aom_core import apply_aom_mitigation


# ==========================================
# 🛡️ 极简 CLIP 包装器 (专门用来骗过 torchattacks)
# torchattacks 需要一个能直接把 [0, 1] 图像变成 Logits 的黑盒
# ==========================================
class CLIPWhiteBox(torch.nn.Module):
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.model = clip_model
        self.text_features = text_features
        # 官方 CLIP 的标配 Normalizer
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, x):
        # 1. 归一化
        x = (x - self.mean) / self.std
        # 2. 提取图像特征
        image_features = self.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # 3. 算 Logits
        return 100.0 * image_features @ self.text_features.t()


def main():
    parser = argparse.ArgumentParser(description='AOM Training-Free Defense')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='DTD')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int)
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    # 攻击参数
    parser.add_argument('--eps', default=4.0, type=float, help='PGD epsilon')
    parser.add_argument('--atk_alpha', default=1.0, type=float, help='PGD alpha')
    parser.add_argument('--steps', type=int, default=7, help='PGD steps')

    # ==========================================
    # 🧠 AOM 专属超参数配置 (对应论文)
    # ==========================================
    parser.add_argument('--aom_noise_std', default=0.1, type=float, help='高斯噪声的强度 (sigma)')
    parser.add_argument('--aom_n_times', default=5, type=int, help='加噪声采样次数 (n)')
    parser.add_argument('--aom_alpha', default=1.5, type=float, help='线性移动步长 (alpha)')

    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"🚀 开始 AOM 测试 | 设备: {device} | 数据集: {args.test_sets}")

    # 1. 加载原生 CLIP
    model, _ = clip.load(args.arch, device=device)
    model.eval()  # 永远锁死
    for param in model.parameters():
        param.requires_grad_(False)

    # 2. 准备固定的文本特征 (标准答案)
    classnames = eval(f"{args.test_sets.lower()}_classes")
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in classnames]
    text_tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 包装模型给攻击器用
    attack_model = CLIPWhiteBox(model, text_features).to(device)
    attack_model.eval()

    # 3. 准备数据加载器
    from torchvision.transforms import InterpolationMode
    data_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor()
    ])

    val_dataset = build_dataset(args.test_sets, data_transform, args.data, mode='test')

    # (调试用) 强行截断，只取前 10 个样本测试流水线。正式测试时请注释掉这一行！
    val_dataset = Subset(val_dataset, list(range(10)))
    print(f"📦 测试样本数量: {len(val_dataset)}")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. 初始化 PGD 攻击器
    atk = torchattacks.PGD(attack_model, eps=args.eps / 255, alpha=args.atk_alpha / 255, steps=args.steps)

    # 记分牌
    top1_adv = AverageMeter('AdvAcc@1', ':6.2f', Summary.AVERAGE)
    top1_aom = AverageMeter('AomAcc@1', ':6.2f', Summary.AVERAGE)

    # 5. 核心测试循环
    for i, (images, target) in enumerate(val_loader):
        images, target = images.to(device), target.to(device)

        # A. 生成对抗样本
        if args.eps > 0:
            adv_images = atk(images, target)
        else:
            adv_images = images

        # B. AOM 特征处理流水线 (纯前向，无梯度)
        with torch.no_grad():
            # 获取归一化后的对抗图像 (x_source)
            adv_images_norm = (adv_images - attack_model.mean) / attack_model.std

            # 1. 获取 Source Feature (f_source)
            adv_features = model.encode_image(adv_images_norm)
            adv_features = adv_features / adv_features.norm(dim=-1, keepdim=True)

            # ==========================================
            # 🚀 AOM: 构造 Anchor Feature (加高斯噪声)
            # ==========================================
            anchor_features_sum = torch.zeros_like(adv_features)

            for _ in range(args.aom_n_times):
                # 生成高斯噪声 (N(0, sigma^2))
                noise = torch.randn_like(adv_images_norm) * args.aom_noise_std

                # 叠加噪声到源图像上
                noisy_images = adv_images_norm + noise

                # 提取特征并归一化
                noisy_feat = model.encode_image(noisy_images)
                noisy_feat = noisy_feat / noisy_feat.norm(dim=-1, keepdim=True)

                # 累加特征
                anchor_features_sum += noisy_feat

            # 求平均得到最终的 f_anchor
            anchor_features = anchor_features_sum / args.aom_n_times

            # ==========================================
            # C. 调用 aom_core.py 执行一步线性移动
            # ==========================================
            aom_features = apply_aom_mitigation(adv_features, anchor_features, alpha=args.aom_alpha)

            # D. 计算攻击后的原准确率 和 AOM 修正后的准确率
            logits_adv = 100.0 * adv_features @ text_features.t()
            logits_aom = 100.0 * aom_features @ text_features.t()

            acc1_adv, _ = accuracy(logits_adv, target, topk=(1, 5))
            acc1_aom, _ = accuracy(logits_aom, target, topk=(1, 5))

            top1_adv.update(acc1_adv[0].item(), images.size(0))
            top1_aom.update(acc1_aom[0].item(), images.size(0))

        print(f"Batch {i + 1}: 无防御对抗准确率: {top1_adv.val:.2f}% | AOM 修复后准确率: {top1_aom.val:.2f}%")

    print("\n" + "=" * 40)
    print(f"🏁 最终评测结果 (Eps: {args.eps}/255, PGD Steps: {args.steps})")
    print(f"🧠 AOM 参数: [n={args.aom_n_times}, sigma={args.aom_noise_std}, alpha={args.aom_alpha}]")
    print(f"❌ 攻击穿透后 Accuracy : {top1_adv.avg:.2f}%")
    print(f"🛡️ AOM 修复后 Accuracy : {top1_aom.avg:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    main()