from typing import Optional

import torch.nn as nn
import torch
from transformers import MusicgenConfig
from flash_attn import flash_attn_with_kvcache


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
        input_pos: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        q: (B, L_q, D)
        k: (B, L_k, D)
        v: (B, L_k, D)
        input_pos: (1,)
        cache_seqlens: (2,)
        cache_batch_idx: (2,)
        """
        B, L_q, D = q.shape
        L_k = k.shape[1]
        q = self.q_proj(q)
        q = q.reshape(B, L_q, self.num_heads, self.head_dim)
        k = self.k_proj(k)
        k = k.reshape(B, L_k, self.num_heads, self.head_dim)
        v = self.v_proj(v)
        v = v.reshape(B, L_k, self.num_heads, self.head_dim)

        k, v = self.cache(k, v, input_pos, cache_batch_idx)

        o = flash_attn_with_kvcache(
            q,
            k,
            v,
            causal=True,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.scale,
            cache_batch_idx=cache_batch_idx,
        )
        o = o.reshape(B, L_q, D)
        o = self.out_proj(o)
        return o

    def set_cache(
        self,
        max_length: int = 1500,
        batch_size: int = 2,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.cache = KVCache(
            head_dim=self.head_dim,
            n_kv_heads=self.num_heads,
            device=device,
            dtype=dtype,
            max_length=max_length,
            batch_size=batch_size,
        )

    def reset_cache(self):
        self.cache.reset_keys_and_values()


class KVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
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
        k_shape = (batch_size, self.max_length, self.n_kv_heads, head_dim)
        v_shape = (batch_size, self.max_length, self.n_kv_heads, head_dim)
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
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        input_pos: torch.Tensor,
        cache_batch_idx: torch.Tensor,
    ):
        """
        keys: (batch_size, 1, n_kv_heads, k_head_dim)
        values: (batch_size, 1, n_kv_heads, v_head_dim)
        input_pos: (batch_size,)
        cache_batch_idx: (batch_size,)
        """
        assert (
            cache_batch_idx.shape[0] == keys.shape[0] == values.shape[0]
        ), f"cache_batch_idx.shape: {cache_batch_idx.shape}, keys.shape: {keys.shape}, values.shape: {values.shape}"
        self.keys[cache_batch_idx[..., None], input_pos[None, ...], ...] = keys
        self.values[cache_batch_idx[..., None], input_pos[None, ...], ...] = values
        return self.keys, self.values
