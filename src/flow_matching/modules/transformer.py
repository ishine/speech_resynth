# Copied and modified from https://github.com/lucidrains/voicebox-pytorch/blob/main/voicebox_pytorch/voicebox_pytorch.py

# MIT License
#
# Copyright (c) 2023 Phil Wang
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

from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .alibi import get_alibi
from .fastspeech.modules import FeedForward
from .norm import AdaptiveRMSNorm


def exists(val):
    return val is not None


class RotaryEmbedding(nn.Module):
    """
    rotary positional embeddings
    https://arxiv.org/abs/2104.09864
    """

    def __init__(self, hidden_size: int, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, t: Union[int, torch.Tensor]):
        if not torch.is_tensor(t):
            t = torch.arange(t, device=self.device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.amp.autocast("cuda", enabled=False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


class Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dropout = dropout

        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, mask=None, rotary_emb=None):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        if mask is not None and mask.ndim != 4:
            mask = mask.unsqueeze(1).unsqueeze(2)

        bsz, heads, q_len, _ = q.shape

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        if mask is not None:
            mask = mask.expand(-1, heads, q_len, -1)

        # alibi = get_alibi(q_len, self.heads, x.device)  # (heads, q_len, q_len)
        # alibi = alibi.unsqueeze(0).expand(bsz, -1, -1, -1)  # (bsz, heads, q_len, q_len)
        # if mask is not None:
        #     mask = mask.masked_fill(~mask, float("-inf"))
        #     mask = alibi + mask
        # else:
        #     mask = alibi

        # Check if there is a compatible device for flash attention
        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        heads: int,
        intermediate_size: int,
        attn_dropout: float,
        ff_dropout: float,
        use_unet_skip_connection: bool,
    ):
        super().__init__()
        assert depth % 2 == 0
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(hidden_size=hidden_size // heads)

        for ind in range(depth):
            layer = ind + 1
            has_skip = use_unet_skip_connection and layer > (depth // 2)

            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(hidden_size * 2, hidden_size, bias=False) if has_skip else None,
                        AdaptiveRMSNorm(hidden_size=hidden_size),
                        Attention(
                            hidden_size=hidden_size,
                            heads=heads,
                            dropout=attn_dropout,
                        ),
                        AdaptiveRMSNorm(hidden_size=hidden_size),
                        FeedForward(hidden_size=hidden_size, intermediate_size=intermediate_size, dropout=ff_dropout),
                    ]
                )
            )

        self.final_norm = nn.RMSNorm(hidden_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, mask=None, adaptive_rmsnorm_cond=None):
        batch, seq_len, *_ = x.shape

        # keep track of skip connections
        skip_connects = []

        # rotary embeddings
        rotary_emb = self.rotary_emb(seq_len)

        # adaptive rmsnorm
        rmsnorm_kwargs = dict()
        if exists(adaptive_rmsnorm_cond):
            rmsnorm_kwargs = dict(condition=adaptive_rmsnorm_cond)

        # going through the attention layers
        for skip_combiner, attn_prenorm, attn, ff_prenorm, ff in self.layers:
            # in the paper, they use a u-net like skip connection
            # unclear how much this helps, as no ablations or further numbers given besides a brief one-two sentence mention

            if not exists(skip_combiner):
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop()
                x = torch.cat((x, skip_connect), dim=-1)
                x = skip_combiner(x)

            attn_input = attn_prenorm(x, **rmsnorm_kwargs)
            x = attn(attn_input, mask=mask, rotary_emb=rotary_emb) + x

            ff_input = ff_prenorm(x, **rmsnorm_kwargs)
            x = ff(ff_input, mask=mask) + x

        return self.final_norm(x)
