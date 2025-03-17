# copied and modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/fastspeech2_conformer/modeling_fastspeech2_conformer.py

# coding=utf-8
# Copyright 2023 The Espnet authors, IMS Toucan authors, and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...configs import ConditionalFlowMatchingConfig


class SiGLU(nn.Module):
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x, gate = x.chunk(2, dim=-2)
        return F.silu(gate) * x


class FeedForward(nn.Module):
    """
    Multi-layered conv1d with a GLU activation function for Transformer block.
    https://arxiv.org/abs/1905.09263
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0, kernel_size: int = 3):
        super().__init__()

        self.conv1 = nn.Conv1d(
            hidden_size, intermediate_size * 2, kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )
        self.glu = SiGLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(intermediate_size, hidden_size, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, hidden_states: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Batch of input tensors.

        Returns:
            `torch.Tensor`: Batch of output tensors `(batch_size, sequence_length, hidden_size)`.
        """
        hidden_states = hidden_states.transpose(-1, 1)

        if mask is not None:
            mask = mask.unsqueeze(1)
            hidden_states = hidden_states.masked_fill(~mask, 0.0)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.glu(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if mask is not None:
            hidden_states = hidden_states.masked_fill(~mask, 0.0)

        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.transpose(-1, 1)
        return hidden_states


class ConditionalFlowMatchingDurationPredictor(nn.Module):
    """
    Duration predictor module.
    https://arxiv.org/abs/1905.09263
    """

    def __init__(self, config: ConditionalFlowMatchingConfig):
        super().__init__()
        self.log_domain_offset = 1.0
        self.conv = nn.Conv1d(config.dim_cond_emb, 1, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.FloatTensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.

        Returns:
            `torch.Tensor`: Batch of predicted durations in log domain `(batch_size, max_text_length)`.

        """
        # (batch_size, input_dim, max_text_length)
        hidden_states = hidden_states.transpose(1, -1)

        # NOTE: calculate in log domain, (batch_size, max_text_length)
        hidden_states = self.conv(hidden_states).squeeze(1)

        if not self.training:
            # NOTE: calculate in linear domain
            hidden_states = torch.clamp(torch.round(hidden_states.exp() - self.log_domain_offset), min=0).long()

        return hidden_states
