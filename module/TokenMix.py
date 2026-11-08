import torch
import torch.nn as nn
class TokenMix(nn.Module):
    def __init__(self, patch_h=2, patch_w=2):
        super(TokenMix, self).__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w

        # 分patch切块
        x = x.reshape(B, C, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 1, 3, 5)  # B, H/ph, W/pw, C, ph, pw
        x = x.reshape(B, (H // ph) * (W // pw), C, ph, pw)  # B, N, C, ph, pw

        # patch内部进行简单shuffle
        idx = torch.randperm(x.size(1))
        x = x[:, idx, :, :, :]

        # 再恢复原状
        x = x.reshape(B, H // ph, W // pw, C, ph, pw)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, C, H, W)

        return x
