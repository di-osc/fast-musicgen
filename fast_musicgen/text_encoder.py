import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, t5_name, input_dim, output_dim):
        super().__init__()
        self._t5, self.tokenizer = T5.from_pretrained(t5_name)
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, text):
        input_ids = self.tokenizer.encode(text)
        x = self._t5.encode(input_ids)
        return self.output_proj(x)


class Tokenizer:
    def __init__(self, config, model_name):
        self._decoder_start_id = config.decoder_start_token_id
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            legacy=False,
            model_max_length=getattr(config, "n_positions", 512),
        )

    @property
    def eos_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def decoder_start_id(self) -> int:
        return self._decoder_start_id

    def encode(self, s: str) -> torch.Tensor:
        return self._tokenizer(
            s,
            return_tensors="pt",
            return_attention_mask=False,
        )["input_ids"]

    def decode(self, t: List[int], with_sep: bool = True) -> str:
        tokens = self._tokenizer.convert_ids_to_tokens(t)
        return "".join(t.replace("▁", " " if with_sep else "") for t in tokens)


def _relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    """
    Adapted from HF Tensorflow:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.int16) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.minimum(
            relative_position, torch.zeros_like(relative_position)
        )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    scale = (num_buckets - max_exact) / np.log(max_distance / max_exact)
    relative_position_if_large = max_exact + (
        torch.log(relative_position.to(torch.float32) / max_exact) * scale
    ).to(torch.int16)
    relative_position_if_large = torch.minimum(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )
    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large
    )
    print(relative_buckets.shape)
    return relative_buckets.to(torch.int32)


class RelativePositionBias(nn.Module):
    def __init__(self, config, bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = getattr(config, "relative_attention_max_distance", 128)
        self.n_heads = config.num_heads
        self.embeddings = nn.Embedding(
            config.relative_attention_num_buckets, config.num_heads
        )

    def forward(self, query_length: int, key_length: int, offset: int = 0):
        """Compute binned relative position bias"""
        context_position = torch.arange(offset, query_length)[:, None]
        memory_position = torch.arange(key_length)[None, :]

        # shape (query_length, key_length)
        relative_position = memory_position - context_position
        relative_position_bucket = _relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        print("relative_position_bucket.shape", relative_position_bucket.shape)
        # shape (query_length, key_length, num_heads)
        values = self.embeddings(relative_position_bucket)

        # shape (num_heads, query_length, key_length)
        return values.permute(2, 0, 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.d_kv * config.num_heads
        self.num_heads = config.num_heads
        self.query_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.key_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.value_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, config.d_model, bias=False)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> [torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).permute(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).permute(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).permute(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            keys = torch.concatenate([key_cache, keys], axis=3)
            values = torch.concatenate([value_cache, values], axis=2)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scores = queries @ keys
        if mask is not None:
            scores = scores + mask.to(scores.dtype)

        scores = torch.softmax(scores.to(torch.float32), axis=-1).to(scores.dtype)
        values_hat = (scores @ values).permute(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(values_hat), (keys, values)


class DenseActivation(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        self.gated = hasattr(config, "feed_forward_proj")
        activation = (
            "relu"
            if not self.gated
            else config.feed_forward_proj.removeprefix("gated-")
        )
        if self.gated:
            self.wi_0 = nn.Linear(config.d_model, mlp_dims, bias=False)
            self.wi_1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        else:
            self.wi = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.wo = nn.Linear(mlp_dims, config.d_model, bias=False)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        if self.gated:
            hidden_act = self.act(self.wi_0(x))
            hidden_linear = self.wi_1(x)
            x = hidden_act * hidden_linear
        else:
            x = self.act(self.wi(x))
        return self.wo(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dense = DenseActivation(config)

    def forward(self, x, mask):
        y = self.ln1(x)
        y, _ = self.attention(y, y, y, mask=mask)
        x = x + y

        y = self.ln2(x)
        y = self.dense(y)
        return x + y


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for i in range(config.num_layers)]
        )
        self.ln = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=True)

    def forward(self, x: torch.Tensor):
        pos_bias = self.relative_attention_bias(x.shape[1], x.shape[1])
        for layer in self.layers:
            x = layer(x, mask=pos_bias)
        return self.ln(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln3 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dense = DenseActivation(config)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        y = self.ln1(x)
        y, cache = self.self_attention(y, y, y, mask, cache)
        x = x + y

        y = self.ln2(x)
        y, _ = self.cross_attention(y, memory, memory, memory_mask)
        x = x + y

        y = self.ln3(x)
        y = self.dense(y)
        x = x + y

        return x, cache


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers = getattr(config, "num_decoder_layers", config.num_layers)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for i in range(n_layers)]
        )
        self.ln = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=False)

    def forward(self, x, memory, mask, memory_mask, cache=None):
        if cache is not None:
            offset = cache[0][0].shape[3]
        else:
            offset = 0
            cache = [None] * len(self.layers)

        T = offset + x.shape[1]
        pos_bias = self.relative_attention_bias(T, T, offset=offset)
        if mask is not None:
            mask += pos_bias
        else:
            mask = pos_bias

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, memory, mask, memory_mask, cache=cache[e])
        x = self.ln(x)

        return x, cache


class OutputHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, inputs):
        return self.linear(inputs)


class T5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if not self.tie_word_embeddings:
            self.lm_head = OutputHead(config)
        self.model_dim = config.d_model

    def encode(self, inputs: torch.Tensor):
        return self.encoder(self.wte(inputs))

    def decode(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        cache=None,
    ):
        inputs = self.wte(inputs)
        T = inputs.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.to(inputs.dtype)
        else:
            mask = None

        y, cache = self.decoder(
            inputs, memory=memory, mask=mask, memory_mask=None, cache=cache
        )
        if not self.tie_word_embeddings:
            y = self.lm_head(y)
        else:
            y *= self.model_dim**-0.5
            y = y @ self.wte.weight.T
        return y, cache

    def forward(
        self,
        inputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
    ):
        return self.decode(decoder_inputs, self.encode(inputs))[0]

    @classmethod
    def sanitize(cls, weights):
        shared_replacement_patterns = [
            (".block.", ".layers."),
            (".k.", ".key_proj."),
            (".o.", ".out_proj."),
            (".q.", ".query_proj."),
            (".v.", ".value_proj."),
            ("shared.", "wte."),
            ("lm_head.", "lm_head.linear."),
            (".layer.0.layer_norm.", ".ln1."),
            (".layer.1.layer_norm.", ".ln2."),
            (".layer.2.layer_norm.", ".ln3."),
            (".final_layer_norm.", ".ln."),
            (
                "layers.0.layer.0.SelfAttention.relative_attention_bias.",
                "relative_attention_bias.embeddings.",
            ),
        ]

        encoder_replacement_patterns = [
            (".layer.0.SelfAttention.", ".attention."),
            (".layer.1.DenseReluDense.", ".dense."),
        ]

        decoder_replacement_patterns = [
            (".layer.0.SelfAttention.", ".self_attention."),
            (".layer.1.EncDecAttention.", ".cross_attention."),
            (".layer.2.DenseReluDense.", ".dense."),
        ]

        ignored_keys = [
            "decoder.layers.0.cross_attention.relative_attention_bias.weight"
        ]

        def replace_key(key: str) -> str:
            for old, new in shared_replacement_patterns:
                key = key.replace(old, new)
            if key.startswith("encoder."):
                for old, new in encoder_replacement_patterns:
                    key = key.replace(old, new)
            elif key.startswith("decoder."):
                for old, new in decoder_replacement_patterns:
                    key = key.replace(old, new)
            return key

        weights = {replace_key(k): v for k, v in weights.items()}
        for key in ignored_keys:
            if key in weights:
                del weights[key]
        return weights

    @classmethod
    def from_pretrained(
        cls, path_or_repo: str, dtype: torch.dtype = torch.bfloat16
    ) -> tuple["T5", Tokenizer]:
        from huggingface_hub import snapshot_download

        path = Path(path_or_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_repo,
                    allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin"],
                )
            )

        with open(path / "config.json", "r") as f:
            config = SimpleNamespace(**json.load(f))
        with torch.device("cpu"):
            model = T5(config)
        weights = torch.load(str(path / "pytorch_model.bin"))
        weights = cls.sanitize(weights)
        weights = {k: v.to(dtype) for k, v in weights.items()}
        model.load_state_dict(weights)
        return model.eval(), Tokenizer(config, "t5-base")
