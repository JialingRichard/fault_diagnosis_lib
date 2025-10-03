__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp




class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x
    


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.PatchTST_backbone import PatchTST_backbone
from ..layers.PatchTST_layers import series_decomp

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
                 mlp_hidden: int = None,       # 默认 = d_model
                 mlp_dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.pooling = pooling
        self.d_model = d_model

        # Backbone (输出 patch token embeddings)
        self.backbone = PatchTST_backbone(
            c_in=input_dim, context_window=time_steps, target_window=0,
            patch_len=patch_len, stride=stride,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
            dropout=dropout, head_type="token",  # 输出 token 表示
            **kwargs
        )

        # Attention pooling 需要的参数
        if pooling == "attention":
            self.attention_pool = nn.Linear(d_model, 1)

        # 分类 head (MLP)
        mlp_hidden = mlp_hidden or d_model
        layers = []
        in_dim = d_model
        for i in range(mlp_layers - 1):
            layers.append(nn.Linear(in_dim, mlp_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(mlp_dropout))
            in_dim = mlp_hidden
        layers.append(nn.Linear(in_dim, output_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)
        Returns:
            out: (B, num_classes)
        """
        # Backbone: (B,L,C) -> (B,C,L) -> (B,N,D)
        x = x.permute(0, 2, 1)
        tokens = self.backbone(x)  # (B, N, d_model)

        # 池化，把 token 序列变成单向量
        if self.pooling == "mean":
            pooled = tokens.mean(dim=1)
        elif self.pooling == "last":
            pooled = tokens[:, -1, :]
        elif self.pooling == "max":
            pooled, _ = tokens.max(dim=1)
        elif self.pooling == "attention":
            attn_weights = torch.softmax(
                self.attention_pool(tokens).squeeze(-1), dim=1
            )  # (B,N)
            pooled = torch.sum(tokens * attn_weights.unsqueeze(-1), dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        # 分类
        return self.classifier(pooled)  # (B,num_classes)
