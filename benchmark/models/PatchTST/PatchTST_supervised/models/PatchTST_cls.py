__all__ = ['PatchTST']

# Cell
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import sys, os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(__file__)
# 切到上一级（PatchTST_supervised）
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

# 可选：恢复 sys.path（保证全局干净）
sys.path.pop(0)


class PatchTSTClassifier(nn.Module):
    """
    PatchTST for time-series classification (seq2one)
    输入: (B, L, C)
    输出: (B, num_classes)
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 time_steps: int,        # 序列长度
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 pooling: str = "mean",        # mean, last, max, attention
                 mlp_layers: int = 2,          # 分类 head 层数
                 mlp_hidden: int = None,       # 默认 = embed_dim
                 mlp_dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.pooling = pooling
        self.d_model = d_model
        self.input_dim = input_dim

        # Backbone (输出 patch token embeddings)
        self.backbone = PatchTST_backbone(
            c_in=input_dim, context_window=time_steps, target_window=0,
            patch_len=patch_len, stride=stride,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
            dropout=dropout, head_type="token",  # 输出 token 表示
            **kwargs
        )

        # token 实际维度 = C * d_model
        embed_dim = input_dim * d_model

        # Attention pooling 需要的参数
        if pooling == "attention":
            self.attention_pool = nn.Linear(embed_dim, 1)

        # 分类 head (MLP)
        mlp_hidden = mlp_hidden or embed_dim
        layers = []
        in_dim = embed_dim
        for i in range(mlp_layers - 1):
            layers.append(nn.Linear(in_dim, mlp_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(mlp_dropout))
            in_dim = mlp_hidden
        layers.append(nn.Linear(in_dim, output_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, L, C)
        Returns:
            out: (B, num_classes)
        """
        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        tokens = self.backbone.forward_tokens(x)  # (B, C, d_model, P)

        # 展平 (B, C, d_model, P) -> (B, P, C*d_model)
        tokens = tokens.flatten(1, 2).permute(0, 2, 1)

        # 池化
        if self.pooling == "mean":
            pooled = tokens.mean(dim=1)
        elif self.pooling == "last":
            pooled = tokens[:, -1, :]
        elif self.pooling == "max":
            pooled, _ = tokens.max(dim=1)
        elif self.pooling == "attention":
            attn_weights = torch.softmax(self.attention_pool(tokens).squeeze(-1), dim=1)
            pooled = torch.sum(tokens * attn_weights.unsqueeze(-1), dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        return self.classifier(pooled)  # (B, num_classes)
