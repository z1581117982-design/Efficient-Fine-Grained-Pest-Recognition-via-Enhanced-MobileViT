import math
from torch import nn
import torch
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型的维度
        self.d_k = d_model // heads  # 每个头的维度
        self.h = heads  # 头的数量

        # 以下三个是线性层，用于处理Q（Query），K（Key），V（Value）
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.out = nn.Linear(d_model, d_model)  # 输出层

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        # torch.matmul是矩阵乘法，用于计算query和key的相似度
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)  # 在第一个维度增加维度
            scores = scores.masked_fill(mask == 0, -1e9)  # 使用mask将不需要关注的位置设置为一个非常小的数

        # 对最后一个维度进行softmax运算，得到权重
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)  # 应用dropout

        output = torch.matmul(scores, v)  # 将权重应用到value上
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)  # 获取batch_size

        # 将Q, K, V通过线性层处理，然后分割成多个头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 转置来获取维度为bs * h * sl * d_model的张量
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 调用attention函数计算输出
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # 重新调整张量的形状，并通过最后一个线性层
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)  # 最终输出
        return output