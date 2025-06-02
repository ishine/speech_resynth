# from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/decoder.py

# MIT License
#
# Copyright (c) 2023 Shivam Mehta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
from torch import nn


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, freq_embed_size: int = 256, scale=1000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
        )
        self.freq_embed_size = freq_embed_size
        self.scale = scale

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
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=timesteps.dtype, device=timesteps.device) / (half - 1)
        )
        args = self.scale * timesteps.unsqueeze(1) * freqs.unsqueeze(0)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        embeddings = self.mlp(embeddings)
        return embeddings
