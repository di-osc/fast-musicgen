from __future__ import annotations
from typing import Optional

import torch.nn as nn
import torch
from transformers import MusicgenConfig


class CausalLM(nn.Module):
    def __init__(self, config: MusicgenConfig):
        super().__init__()
        self.config = config
        self.num_codebooks = config.decoder.num_codebooks
        self.codebook_size = config.audio_encoder.codebook_size
        self.bos_token_id = config.decoder.bos_token_id
        self.hidden_size = config.decoder.hidden_size
        self.num_attention_heads = config.decoder.num_attention_heads

        self.embed = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size + 1, self.hidden_size)
                for _ in range(self.num_codebooks)
            ]
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.decoder.num_hidden_layers)]
        )
        self.out_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.codebook_size, bias=False)
                for _ in range(self.num_codebooks)
            ]
        )

    def forward(
        self,
        audio_x: torch.Tensor,
        text_x: torch.Tensor,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        pass

    def generate(self, text_x: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        pass


class TransformerBlock(nn.Module):
    def __init__(self, config: MusicgenConfig):
        super().__init__()
        pass


class KVCache:
    def __init__(self, head_dim, n_kv_heads):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256
    
    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = torch.zeros(k_shape, dtype=keys.dtype)
            new_v = torch.zeros(v_shape, dtype=values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = torch.concatenate([self.keys, new_k], dim=2)
                self.values = torch.concatenate([self.values, new_v], dim=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
