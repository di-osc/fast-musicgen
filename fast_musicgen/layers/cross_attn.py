import torch.nn as nn
import torch
from transformers import MusicgenConfig


class CrossAttention(nn.Module):
    def __init__(self, config: MusicgenConfig):
        super().__init__()
        self.q_proj = nn.Linear(
            config.decoder.hidden_size, config.decoder.hidden_size, bias=False
        )
        self.k_proj = nn.Linear(
            config.decoder.hidden_size, config.decoder.hidden_size, bias=False
        )
        self.v_proj = nn.Linear(
            config.decoder.hidden_size, config.decoder.hidden_size, bias=False
        )
        self.out_proj = nn.Linear(
            config.decoder.hidden_size, config.decoder.hidden_size, bias=False
        )
        self.num_heads = config.decoder.num_attention_heads
        self.head_dim = config.decoder.hidden_size // config.decoder.num_attention_heads
        self.scale = self.head_dim**-0.5

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        B, L_q, D = q.shape
        L_k = k.shape[1]

        q = self.q_proj(q)
        q = q.reshape(B, L_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k)
        k = k.reshape(B, L_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v)
        v = v.reshape(B, L_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        o = nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, scale=self.scale
        )
        o = o.permute(0, 2, 1, 3).reshape(B, L_q, D)
        o = self.out_proj(o)
        return o
