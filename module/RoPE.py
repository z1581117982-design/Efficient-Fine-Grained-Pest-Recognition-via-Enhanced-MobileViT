import torch
import torch.nn as nn

def rotate_half(x):
    """向量后半部分旋转"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

def precompute_freqs(dim, max_len, theta=10000.0):
    """预计算旋转频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    t = torch.arange(max_len, device=freqs.device)
    return torch.outer(t, freqs)


class RotaryAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # 预计算旋转参数
        self.max_len = 4096
        self.freqs = precompute_freqs(self.head_dim, self.max_len)

    def forward(self, x, positions):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim)

        # 获取旋转参数
        cos = torch.cos(self.freqs[positions])
        sin = torch.sin(self.freqs[positions])

        # 应用旋转编码
        q_rot, k_rot = apply_rotary_pos_emb(
            qkv[:, :, 0],
            qkv[:, :, 1],
            cos, sin
        )

        # 标准注意力计算
        attn = (q_rot @ k_rot.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        # 输出投影
        output = (attn @ qkv[:, :, 2]).transpose(1, 2).reshape(B, L, self.dim)
        return self.proj(output)
