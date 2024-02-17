import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, showattn=False):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key))]
        value = value.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        
        if showattn:
            return torch.mean(x, -3), self.attn

        # 3) "Concat" using a view and apply a final linear.
        return torch.mean(x, -3)
    
class ScoreMap_embedding_concat(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.layer_norm = nn.LayerNorm(h*h)
        self.fcl1 = nn.Linear(2, 256)
        self.fcl2 = nn.Linear(256, 1)

        self.fcl1.weight.data = torch.abs(self.fcl1.weight.data)
        self.fcl2.weight.data = torch.abs(self.fcl2.weight.data)

        self.fcl1.bias.data = torch.tensor([0.], dtype=torch.float)
        self.fcl2.bias.data = torch.tensor([0.], dtype=torch.float)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_copy = x.clone()
        x = self.layer_norm(x)
        x = torch.cat((x_copy, x), dim=-2)
        x = x.permute(0, 2, 1)
        x = self.fcl1(x)
        x = self.fcl2(x)
        x = self.sigmoid(x)
        return x.permute(0, 2, 1)
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query, key, value):
        query = query.permute(0, 2, 1)
        key = key.permute(0, 2, 1)
        value = value.permute(0, 2, 1)

        B, N, C = query.shape
        q = self.wq(query).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(key).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(value).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BHN(C/H) @ BH(C/H)N -> BHNN
        attn = attn.softmax(dim=-1)

        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, C) # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        return x.permute(0, 2, 1)
    
    def embedding_value(self, value):
        value = value.permute(0, 2, 1)
        x = self.wv(value)
        x = self.proj(x)
        return x.permute(0, 2, 1)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])