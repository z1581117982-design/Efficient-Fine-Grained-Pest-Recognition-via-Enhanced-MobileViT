import os

# 关闭无关的 TensorFlow 输出
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import torch
import torch.optim as optim
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from DataLoader.IP102_dataset import IP102Dataset
from model import mobile_vit_small as create_model
from utils import train_one_epoch, evaluate
from torch.optim.lr_scheduler import CosineAnnealingLR

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(args, img_size=224):

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = IP102Dataset(
        root_dir=args.train_path,
        label_json=args.label_json,
        transform=data_transform["train"]
    )
    val_dataset = IP102Dataset(
        root_dir=args.val_path,
        label_json=args.label_json,
        transform=data_transform["val"]
    )

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers per process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn
    )

    return train_loader, val_loader

# 加载模型
def build_model(args, device):

    model = create_model(num_classes=args.num_classes).to(device)   # 实例化模型

    if args.weights:
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        weights_dict = torch.load(args.weights, map_location=device, weights_only=False,)     # 加载预训练权重
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    # 冻结除分类器外的权重
    if args.freeze_layers:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad_(False)
            else:
                print(f"training {name}")

    return model


class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        """
        参数：
        patience: 如果验证集精度在多少个epoch内没有提升，提前停止训练
        delta: 用来判断精度提升的阈值，只有超过此值才算提升
        """
        self.patience = patience
        self.delta = delta
        self.best_val_acc = None
        self.epochs_without_improvement = 0

    def should_stop(self, val_acc):
        """判断是否停止训练"""
        if self.best_val_acc is None or val_acc - self.best_val_acc > self.delta:
            self.best_val_acc = val_acc
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            print(f"Early stopping")
            return True
        return False


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs("./weights", exist_ok=True)
    tb_writer = SummaryWriter()

    # 数据加载
    train_loader, val_loader = prepare_data(args)

    # 模型构建
    model = build_model(args, device)

    # 优化器 & 学习率调度器
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 早停机制
    early_stopping = EarlyStopping(patience=10, delta=0.001)

    best_acc = 0.0

    for epoch in range(args.epochs):
        # 训练 & 验证
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, epoch)

        val_loss, val_acc = evaluate(
            model, val_loader, device, epoch)

        # 更新学习率
        scheduler.step()

        # 日志记录
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # 模型保存
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights/model_SimAM_CdAtt_IP102.pth")

        # 早停检查
        if early_stopping.should_stop(val_acc):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=102)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_path', type=str,
                        default="data/IP102/train",
                        help="Path to training set")
    # parser.add_argument('--train-split-file', type=str, default="data/IP102/train.txt")
    parser.add_argument('--val_path', type=str,
                        default="data/IP102/val",
                        help="Path to validation set")
    # parser.add_argument('--val-split-file', type=str, default="data/IP102_origin/val.txt")
    parser.add_argument('--label-json', type=str, default="data/IP102/labels.json")

    parser.add_argument('--weights', type=str, default='type/mobilevit_s.pt',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0',
                        help='device id (e.g., "cuda:0" or "cpu")')

    opt = parser.parse_args()
    main(opt)
