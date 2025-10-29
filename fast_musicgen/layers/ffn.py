import torch.nn as nn
import torch
from transformers import MusicgenConfig


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
