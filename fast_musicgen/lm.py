from __future__ import annotations
from typing import Optional, List, Dict
from pathlib import Path

import torch.nn as nn
import torch
from transformers import MusicgenConfig
from tqdm import tqdm


class CausalLM(nn.Module):
    def __init__(self, config: MusicgenConfig, device: str):
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
        self.layers: List[TransformerBlock] = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.decoder.num_hidden_layers)]
        )
        self.out_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.codebook_size, bias=False)
                for _ in range(self.num_codebooks)
            ]
        )
        self.device = device

    def forward(
        self,
        audio_x: torch.Tensor,
        text_x: torch.Tensor,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        audio_x: (2, 1, num_codebooks)
        text_x: (2, seq_len, hidden_size)
        """
        if cache is None:
            cache = [None] * len(self.layers)
        audio_x = sum(
            [self.embed[k](audio_x[..., k]) for k in range(self.num_codebooks)]
        )
        offset = cache[0].offset if cache[0] is not None else 0
        pos_emb = self.create_sin_embedding(offset, self.hidden_size)
        audio_x += pos_emb.to(audio_x.dtype)

        for layer, cache in zip(self.layers, cache):
            audio_x = layer(audio_x=audio_x, text_x=text_x, cache=cache)

        audio_x = self.out_norm(audio_x)
        audio_x = torch.stack(
            [self.heads[k](audio_x) for k in range(self.num_codebooks)], dim=-1
        )
        return audio_x

    @torch.inference_mode()
    def generate(
        self,
        text_x: torch.Tensor,
        max_steps: int = 150,
        temperature: float = 1.0,
        top_k: int = 250,
        guidance_coef: float = 3.0,
    ) -> torch.Tensor:
        """generates a waveform conditioned on the text

        Args:
            text_x (torch.Tensor): (1, seq_len, hidden_size)
            max_steps (int, optional): maximum number of steps to generate. Defaults to 150.
            temperature (float, optional): temperature for sampling. Defaults to 1.0.
            top_k (int, optional): top-k sampling. Defaults to 250.
            guidance_coef (float, optional): guidance coefficient for guidance sampling. Defaults to 3.0.

        Returns:
            torch.Tensor: (1, max_steps, num_codebooks)
        """
        text_x = text_x.to(self.device)
        # Assuming no audio prompt we start with all bos token for the codebooks
        audio_shape = (1, max_steps + 1, self.num_codebooks)
        audio_seq = torch.full(audio_shape, self.bos_token_id).to(self.device)

        # Compute conditional and unconditional logits in one batch
        text_tokens = torch.concatenate(
            [text_x, torch.zeros_like(text_x).to(self.device)], dim=0
        )
        head_dim = self.hidden_size // self.num_attention_heads
        cache = [
            KVCache(head_dim, self.num_attention_heads, self.device)
            for _ in range(len(self.layers))
        ]
        for offset in tqdm(range(max_steps)):
            audio_input = torch.tile(audio_seq[:, offset : offset + 1], [2, 1, 1])
            audio_logits = self(audio_input, text_tokens, cache)
            cond_logits, uncond_logits = audio_logits[:1], audio_logits[1:2]
            audio_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_coef
            audio_tokens = self.top_k_sampling(
                logits=audio_logits, top_k=top_k, temperature=temperature, dim=-2
            )
            # "delay" pattern
            audio_tokens[..., offset + 1 :] = self.bos_token_id
            audio_tokens[..., : -max_steps + offset] = self.bos_token_id
            audio_seq[:, offset + 1 : offset + 2] = audio_tokens

        # Undo delay
        for i in range(self.num_codebooks):
            audio_seq[:, : -self.num_codebooks, i] = audio_seq[
                :, i : -self.num_codebooks + i, i
            ]

        audio_seq = audio_seq[:, 1 : -self.num_codebooks + 1]
        audio_seq = torch.swapaxes(audio_seq, -1, -2)[:, torch.newaxis]
        return audio_seq

    def create_sin_embedding(
        self, positions: torch.Tensor, dim: int, max_period: float = 10000
    ) -> torch.Tensor:
        assert dim % 2 == 0
        half_dim = dim // 2
        adim = torch.arange(half_dim).reshape(1, 1, -1)
        phase = positions / (max_period ** (adim / (half_dim - 1)))
        return torch.concatenate([torch.cos(phase), torch.sin(phase)], dim=-1).to(
            self.device
        )

    def top_k_sampling(
        self, logits: torch.Tensor, top_k: int, temperature: float, dim: int = -2
    ) -> torch.Tensor:
        values, indices = logits.topk(k=top_k, dim=dim)
        logits = torch.full_like(logits, -float("inf")).scatter_(
            dim=dim, index=indices, src=values
        )
        probs = logits.float().div_(temperature).softmax(dim=dim)
        tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=dim, keepdim=True)
        return tokens

    @classmethod
    def sanitize(cls, states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_states = {}
        ignore_keys = ["condition_provider"]
        for k, arr in states.items():
            if any(key in k for key in ignore_keys):
                continue
            if k.startswith("transformer."):
                k = k[len("transformer.") :]
            if "emb" in k:
                k = k.replace("emb", "embed")
            if "cross_attention" in k:
                k = k.replace("cross_attention", "cross_attn")
            if "norm1" in k:
                k = k.replace("norm1", "self_attn_norm")
            if "norm_cross" in k:
                k = k.replace("norm_cross", "cross_attn_norm")
            if "norm2" in k:
                k = k.replace("norm2", "ffn_norm")
            if "linears" in k:
                k = k.replace("linears", "heads")
            if "linear1" in k:
                k = k.replace("linear1", "ffn.up_proj")
            if "linear2" in k:
                k = k.replace("linear2", "ffn.down_proj")
            if "in_proj_weight" in k:
                dim = arr.shape[0] // 3
                name = "in_proj_weight"
                new_states[k.replace(name, "q_proj.weight")] = arr[:dim]
                new_states[k.replace(name, "k_proj.weight")] = arr[dim : dim * 2]
                new_states[k.replace(name, "v_proj.weight")] = arr[dim * 2 :]
                continue
            new_states[k] = arr
        return new_states

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, device: str | None = None) -> CausalLM:
        if device is None:
            if torch.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        config = MusicgenConfig.from_pretrained(ckpt_dir)
        with torch.device("meta"):
            model = CausalLM(config, device=device)
        states = torch.load(Path(ckpt_dir) / "state_dict.bin", weights_only=True)[
            "best_state"
        ]
        new_states = cls.sanitize(states)
        model.load_state_dict(new_states, assign=True)
        return model.eval().to(device)


class TransformerBlock(nn.Module):
    def __init__(self, config: MusicgenConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.self_attn_norm = nn.LayerNorm(config.decoder.hidden_size, eps=1e-5)
        self.cross_attn = MultiHeadAttention(config)
        self.cross_attn_norm = nn.LayerNorm(config.decoder.hidden_size, eps=1e-5)
        self.ffn = FeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.decoder.hidden_size, eps=1e-5)

    def forward(
        self,
        audio_x: torch.Tensor,
        text_x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # self-attention and residual connection
        audio_xn = self.self_attn_norm(audio_x)
        audio_x += self.self_attn(audio_xn, audio_xn, audio_xn, mask, cache)

        # cross-attention and residual connection
        audio_xn = self.cross_attn_norm(audio_x)
        audio_x += self.cross_attn(audio_xn, text_x, text_x, mask)

        # feed-forward and residual connection
        audio_xn = self.ffn_norm(audio_x)
        audio_x += self.ffn(audio_xn)
        return audio_x


class MultiHeadAttention(nn.Module):
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
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        B, L_q, D = q.shape
        L_k = k.shape[1]

        q = self.q_proj(q)
        q = q.reshape(B, L_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k)
        k = k.reshape(B, L_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v)
        v = v.reshape(B, L_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        o = nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=mask, scale=self.scale
        )
        o = o.permute(0, 2, 1, 3).reshape(B, L_q, D)
        o = self.out_proj(o)
        return o


class FeedForward(nn.Module):
    def __init__(self, config: MusicgenConfig):
        super().__init__()
        self.up_proj = nn.Linear(
            config.decoder.hidden_size, config.decoder.ffn_dim, bias=False
        )
        self.down_proj = nn.Linear(
            config.decoder.ffn_dim, config.decoder.hidden_size, bias=False
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class KVCache:
    def __init__(self, head_dim, n_kv_heads, device: str, num_steps: int = 256):
        super().__init__()
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
        self.step = num_steps
        self.device = device

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = torch.zeros(k_shape, dtype=keys.dtype, device=self.device)
            new_v = torch.zeros(v_shape, dtype=values.dtype, device=self.device)
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
