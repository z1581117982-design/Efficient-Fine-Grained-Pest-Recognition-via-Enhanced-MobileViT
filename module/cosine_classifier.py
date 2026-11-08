# module/cosine_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x):
        # x: [B, C]
        x = F.normalize(x, dim=1)                          # L2 normalize features
        weight = F.normalize(self.weight, dim=1)           # L2 normalize weights
        cosine = F.linear(x, weight)                       # Cosine similarity
        logits = self.scale * cosine                       # Optional scale factor
        return logits
