import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class IP102Dataset(Dataset):
    """
    加载已经重分类为 IP102_split/train/类别名/xxx.jpg 的图像分类数据集，
    类别名由 labels.json 决定，映射编号从1开始。
    """

    def __init__(self, root_dir, label_json, transform=None):
        """
        :param root_dir: 数据根目录，如 './IP102_split/train'
        :param label_json: 类别名映射文件路径，如 './labels.json'
        :param transform: torchvision transforms，用于图像增强
        """
        self.root_dir = root_dir
        self.transform = transform

        # === 读取类别映射：编号（字符串） -> 类别名 ===
        with open(label_json, 'r') as f:
            idx_to_class = json.load(f)

        # === 构造类别名列表和映射（编号从 1 开始）===
        self.class_names = [idx_to_class[str(i + 1)] for i in range(len(idx_to_class))]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # === 收集所有图像路径和对应标签 ===
        self.image_paths = []
        self.labels = []

        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[跳过] 目录不存在: {class_dir}")
                continue

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, fname)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels
