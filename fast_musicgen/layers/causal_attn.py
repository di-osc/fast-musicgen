import torch.nn as nn
import torch
from transformers import MusicgenConfig
from typing import Optional


class CausalAttention(nn.Module):
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
        self.cache: Optional[KVCache] = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L_q, D = q.shape
        L_k = k.shape[1]

        q = self.q_proj(q)
        q = q.reshape(B, L_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k)
        k = k.reshape(B, L_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v)
        v = v.reshape(B, L_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.cache is not None:
            assert input_pos is not None
            k, v = self.cache(k, v, input_pos)

        o = nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=mask, scale=self.scale
        )
        o = o.permute(0, 2, 1, 3).reshape(B, L_q, D)
        o = self.out_proj(o)
        return o

    def set_cache(
        self,
        max_length: int = 1500,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.cache = KVCache(
            head_dim=self.head_dim,
            n_kv_heads=self.num_heads,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )

    def reset_cache(self):
        self.cache.reset_keys_and_values()


class KVCache(nn.Module):
    def __init__(
        self,
        head_dim: int,
        n_kv_heads: int,
        device: str,
        dtype: torch.dtype = torch.float16,
        max_length: int = 1500,
    ):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.device = device
        self.max_length = max_length
        k_shape = (2, self.n_kv_heads, self.max_length, head_dim)
        v_shape = (2, self.n_kv_heads, self.max_length, head_dim)
        self.register_buffer(
            "keys",
            torch.zeros(k_shape, dtype=dtype, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "values",
            torch.zeros(v_shape, dtype=dtype, device=self.device),
            persistent=False,
        )

    def reset_keys_and_values(self):
        self.keys.fill_(0)
        self.values.fill_(0)

    def forward(
        self, keys: torch.Tensor, values: torch.Tensor, input_pos: torch.Tensor
    ):
        """
        keys: (2, n_kv_heads, 1, k_head_dim)
        values: (2, n_kv_heads, 1, v_head_dim)
        input_pos: (1,)
        """
        assert keys.shape[2] == 1
        assert values.shape[2] == 1
        assert input_pos.shape[0] == 1
        self.keys[..., input_pos, :] = keys
        self.values[..., input_pos, :] = values
        return self.keys, self.values
