import os
import json
import time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from DataLoader.IP102_dataset import IP102Dataset
from torch.utils.data import DataLoader
from model_lib.model_SimAM_DFc import mobile_vit_small as create_model
from torchvision import transforms
import pandas as pd
import random


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_classification_report(y_true, y_pred, class_names, report_path):
    """将分类报告保存为 CSV 文件"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")


def save_confusion_matrix(y_true, y_pred, class_names, cm_path):
    """保存混淆矩阵为 PNG 文件"""
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(32, 24))  # 增大图像尺寸
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True)

    plt.xticks(rotation=90)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")


def main():
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = torchvision.transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_root = "/root/autodl-tmp/MobileVit/data/IP102/test"
    label_json = '/root/autodl-tmp/MobileVit/data/IP102/labels.json'
    assert os.path.exists(test_root), f"Test data not found at: {test_root}"

    # 加载数据和类别索引
    test_dataset = IP102Dataset(
        root_dir=test_root,
        label_json=label_json,
        transform=data_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=test_dataset.collate_fn)

    # 加载可读的类别名称
    assert os.path.exists(label_json), f"{label_json} not found."
    with open(label_json, 'r') as f:
        idx_to_class = json.load(f)
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    # 创建并加载模型
    model = create_model(num_classes=len(idx_to_class)).to(device)
    model.load_state_dict(torch.load("/root/autodl-tmp/MobileVit/weights/model_base_IP102.pth", map_location=device, weights_only=False))
    model.eval()

    y_true, y_pred = [], []
    misclassified_samples = []

    total_time = 0
    total_images = 0
    batch_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            start_time = time.time()

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # 错误样本保存
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified_samples.append((images[i].cpu(), labels[i].item(), preds[i].item()))

            # 统计预测结果
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            elapsed = time.time() - start_time
            batch_times.append(elapsed)
            total_time += elapsed
            total_images += images.size(0)

    # ====== 输出结果 ======
    avg_infer_time = total_time / len(test_loader)
    throughput = total_images / total_time
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))

    class_names = [idx_to_class[i + 1] for i in range(len(idx_to_class))]

    # 保存分类报告
    save_classification_report(y_true, y_pred, class_names, "test_classification_report.csv")

    # 保存混淆矩阵
    save_confusion_matrix(y_true, y_pred, class_names, "test_confusion_matrix.png")

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\n平均推理时间: {avg_infer_time:.4f} 秒/批次")
    print(f"吞吐量: {throughput:.2f} 张/秒")
    print(f"Top-1 准确率: {accuracy:.2%}")
    print(f"宏 F1-score: {macro_f1:.2%}")
    print(f"加权 F1-score: {weighted_f1:.2%}")


if __name__ == '__main__':
    main()