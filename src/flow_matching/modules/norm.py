# from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py

# MIT License
#
# Copyright (c) 2020 Phil Wang
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

import torch.nn.functional as F
from einops import rearrange
from torch import nn


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = hidden_size**0.5
        self.to_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.to_weight.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")

        normed = F.normalize(x, dim=-1)
        gamma = self.to_weight(condition)
        return normed * self.scale * (gamma + 1.0)


class AdaLNZero(nn.Module):
    """
    https://arxiv.org/abs/2212.09748
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.to_weight = nn.Linear(hidden_size, hidden_size)

        nn.init.zeros_(self.to_weight.weight)
        nn.init.zero_(self.to_weight.bias)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")

        gamma = self.to_weight(condition)
        return x * gamma
