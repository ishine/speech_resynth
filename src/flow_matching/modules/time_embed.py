import math

import torch
from torch import nn


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
        )
        self.freq_embed_size = freq_embed_size

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                diffusion timesteps.
        Returns:
            embedding (`torch.Tensor` of shape `(batch_size, hidden_size)`):
                condition for adaptive norm layers.
        """
        half = self.freq_embed_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=timesteps.dtype, device=timesteps.device) / half)
        args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        embeddings = self.mlp(embeddings)
        return embeddings
