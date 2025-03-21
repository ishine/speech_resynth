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

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig, PreTrainedModel
from transformers.models.fastspeech2_conformer.modeling_fastspeech2_conformer import length_regulator

from ..bigvgan.bigvgan import BigVGAN, BigVGanConfig
from ..hifigan.data import dynamic_range_compression_torch
from .configs import (
    ConditionalFlowMatchingConfig,
    ConditionalFlowMatchingWithBigVGanConfig,
    ConditionalFlowMatchingWithHifiGanConfig,
)
from .modules.fastspeech.modules import ConditionalFlowMatchingDurationPredictor
from .modules.fourier_embed import RandomFourierEmbed
from .modules.transformer import ConvPositionEmbed, Transformer


class ConditionalFlowMatchingModel(PreTrainedModel):
    config_class = ConditionalFlowMatchingConfig

    def __init__(self, config: ConditionalFlowMatchingConfig, embedding: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.config = config

        self.time_cond_mlp = nn.Sequential(
            RandomFourierEmbed(config.hidden_size), nn.Linear(config.hidden_size + 1, config.hidden_size), nn.SiLU()
        )
        self.to_cond_emb = (
            nn.Embedding(config.vocab_size + 1, config.dim_cond_emb, padding_idx=0) if embedding is None else embedding
        )
        self.to_embed = nn.Linear(config.dim_in + config.dim_cond_emb, config.hidden_size)
        self.conv_embed = ConvPositionEmbed(
            hidden_size=config.hidden_size,
            kernel_size=config.conv_pos_embed_kernel_size,
            groups=config.conv_pos_embed_groups,
        )

        self.transformer = Transformer(
            hidden_size=config.hidden_size,
            depth=config.depth,
            heads=config.heads,
            intermediate_size=config.intermediate_size,
            ff_dropout=config.ff_dropout,
            use_unet_skip_connection=config.use_unet_skip_connection,
            attn_dropout=config.attn_dropout,
        )

        self.to_pred = nn.Linear(config.hidden_size, config.dim_in, bias=False)
        self.duration_predictor = ConditionalFlowMatchingDurationPredictor(config) if config.predict_duration else None

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.LongTensor,
        spectrogram_labels: torch.FloatTensor,
        duration_labels: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`):
                Batch of padded target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Batch of padded durations.
        """
        mask = (spectrogram_labels != -100).any(dim=-1)
        batch, seq_len, _ = spectrogram_labels.shape
        spectrogram_labels = (spectrogram_labels - self.config.mean) / self.config.std

        # main conditional flow logic is below
        x0 = torch.randn_like(spectrogram_labels)
        times = torch.rand((batch,), device=self.device)
        t = times.unsqueeze(1).unsqueeze(2)
        xt = (1 - t) * x0 + t * spectrogram_labels
        ut = spectrogram_labels - x0

        # phoneme or semantic conditioning embedding
        hidden_states = self.to_cond_emb(input_ids)

        # forward duration predictor
        duration_loss = 0
        if self.config.predict_duration:
            duration_predictions = self.duration_predictor(hidden_states)
            # use groundtruth in training
            hidden_states = length_regulator(hidden_states, duration_labels)

            attention_mask = input_ids.ne(0)
            duration_predictions = duration_predictions.masked_select(attention_mask)
            duration_labels = duration_labels.masked_select(attention_mask)
            duration_labels = torch.log(duration_labels.float() + self.duration_predictor.log_domain_offset)
            duration_loss = F.mse_loss(duration_predictions, duration_labels)

        hidden_states = torch.cat([xt, hidden_states], dim=-1)

        x = self.to_embed(hidden_states)
        x = self.conv_embed(x, mask=mask) + x

        time_emb = self.time_cond_mlp(times)

        # attend
        x = self.transformer(x, mask=mask, adaptive_rmsnorm_cond=time_emb)
        x = self.to_pred(x)

        return F.mse_loss(x[mask], ut[mask]) + duration_loss

    @torch.inference_mode()
    def synthesize(
        self,
        input_ids: torch.LongTensor,
        dt: float = 0.1,
        truncation_value: Optional[float] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            dt (`float`, defaults to 0.1):
                Step size for the ordinary differential equation (ODE).
            truncation_value (`float`, *optional*, defaults to `None`):
                Truncation value of a prior sample x0~N(0, 1).
                https://arxiv.org/abs/1809.11096
        Returns:
            x1 (`torch.FloatTensor` of shape `(batch_size, sequence_length, dim_in)`):
                Synthesized log mel-spectrograms.
        """
        mask = input_ids.ne(0)

        hidden_states = self.to_cond_emb(input_ids)

        # forward duration predictor
        if self.config.predict_duration:
            duration_predictions = self.duration_predictor(hidden_states)
            duration_predictions = duration_predictions.masked_fill(~mask, 0.0)
            hidden_states = length_regulator(hidden_states, duration_predictions)

            # update mask
            lengths = duration_predictions.sum(dim=1, keepdim=True)  # (bsz, 1)
            mask = torch.arange(0, lengths.max(), device=lengths.device).unsqueeze(0) < lengths

        bsz, seq_len, _ = hidden_states.shape

        xt = torch.randn(bsz, seq_len, self.config.dim_in, device=hidden_states.device)
        if truncation_value is not None:
            xt = torch.clamp(xt, -truncation_value, truncation_value)

        for t in torch.arange(0, 1, dt, device=self.device):
            # concat source signal, semantic / phoneme conditioning embed, and conditioning
            # and project
            x = torch.cat([xt, hidden_states], dim=-1)
            x = self.to_embed(x)
            x = self.conv_embed(x, mask=mask) + x

            time_emb = self.time_cond_mlp(t.unsqueeze(0).expand(bsz))

            # attend
            x = self.transformer(x, mask=mask, adaptive_rmsnorm_cond=time_emb)
            vt = self.to_pred(x)
            xt = xt + vt * dt

        x1 = xt * self.config.std + self.config.mean
        x1[~mask] = dynamic_range_compression_torch(torch.tensor(0))

        return x1


class ConditionalFlowMatchingWithHifiGan(PreTrainedModel):
    config_class = ConditionalFlowMatchingWithHifiGanConfig

    def __init__(self, config: ConditionalFlowMatchingWithHifiGanConfig):
        super().__init__(config)
        self.model = ConditionalFlowMatchingModel(config.model_config)
        self.vocoder = FastSpeech2ConformerHifiGan(config.vocoder_config)

    @classmethod
    def load_pretrained(cls, model_path, vocoder_path) -> "ConditionalFlowMatchingWithHifiGan":
        model_config = ConditionalFlowMatchingConfig.from_pretrained(model_path)
        vocoder_config = FastSpeech2ConformerHifiGanConfig.from_pretrained(vocoder_path)
        config = ConditionalFlowMatchingWithHifiGanConfig(model_config.to_dict(), vocoder_config.to_dict())

        model = cls(config)
        model.model = ConditionalFlowMatchingModel.from_pretrained(model_path)
        model.vocoder = FastSpeech2ConformerHifiGan.from_pretrained(vocoder_path)
        return model

    def _get_waveform_lengths(self, spectrogram_lengths):
        def _conv_out_len(input_len, kernel_size, stride, padding):
            # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            return (input_len - 1) * stride - 2 * padding + kernel_size

        for kernel_size, stride in zip(
            self.config.vocoder_config.upsample_kernel_sizes, self.config.vocoder_config.upsample_rates
        ):
            spectrogram_lengths = _conv_out_len(spectrogram_lengths, kernel_size, stride, (kernel_size - stride) // 2)

        return spectrogram_lengths

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        dt: float = 0.1,
        truncation_value: Optional[float] = None,
    ) -> List[torch.FloatTensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            dt (`float`, defaults to 0.1):
                Step size for the ordinary differential equation (ODE).
            truncation_value (`float`, *optional*, defaults to `None`):
                Truncation value of a prior sample x0~N(0, 1).
                https://arxiv.org/abs/1809.11096
        Returns:
            waveform (`list` of `torch.FloatTensor` of shape `(1, (sequence_length - 1) * 320 + 400)`):
                Synthesized waveforms.
        """
        spectrogram = self.model.synthesize(input_ids, dt, truncation_value)

        pad_value = dynamic_range_compression_torch(torch.tensor(0))
        mask = spectrogram.ne(pad_value).all(dim=2)
        spectrogram_lengths = mask.sum(dim=1)
        waveform_lengths = self._get_waveform_lengths(spectrogram_lengths)

        waveform = self.vocoder(spectrogram)

        outputs = []
        for output, length in zip(waveform, waveform_lengths):
            outputs.append(output[:length].unsqueeze(0))

        return outputs


class ConditionalFlowMatchingWithBigVGan(PreTrainedModel):
    config_class = ConditionalFlowMatchingWithBigVGanConfig

    def __init__(self, config: ConditionalFlowMatchingWithBigVGanConfig, use_cuda_kernel=False):
        super().__init__(config)
        self.model = ConditionalFlowMatchingModel(config.model_config)
        self.vocoder = BigVGAN(config.vocoder_config, use_cuda_kernel=False)

    @classmethod
    def load_pretrained(cls, model_path, vocoder_path) -> "ConditionalFlowMatchingWithBigVGan":
        model_config = ConditionalFlowMatchingConfig.from_pretrained(model_path)
        vocoder_config = BigVGanConfig.from_pretrained(vocoder_path)
        config = ConditionalFlowMatchingWithBigVGanConfig(model_config.to_dict(), vocoder_config.to_dict())

        model = cls(config)
        model.model = ConditionalFlowMatchingModel.from_pretrained(model_path)
        model.vocoder = BigVGAN.from_pretrained(vocoder_path)
        return model

    def _get_waveform_lengths(self, spectrogram_lengths):
        def _conv_out_len(input_len, kernel_size, stride, padding):
            # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            return (input_len - 1) * stride - 2 * padding + kernel_size

        for kernel_size, stride in zip(
            self.config.vocoder_config.upsample_kernel_sizes, self.config.vocoder_config.upsample_rates
        ):
            spectrogram_lengths = _conv_out_len(spectrogram_lengths, kernel_size, stride, (kernel_size - stride) // 2)

        return spectrogram_lengths

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        dt: float = 0.1,
        truncation_value: Optional[float] = None,
    ) -> List[torch.FloatTensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            dt (`float`, defaults to 0.1):
                Step size for the ordinary differential equation (ODE).
            truncation_value (`float`, *optional*, defaults to `None`):
                Truncation value of a prior sample x0~N(0, 1).
                https://arxiv.org/abs/1809.11096
        Returns:
            waveform (`list` of `torch.FloatTensor` of shape `(1, (sequence_length - 1) * 320 + 400)`):
                Synthesized waveforms.
        """
        spectrogram = self.model.synthesize(input_ids, dt, truncation_value)

        pad_value = dynamic_range_compression_torch(torch.tensor(0))
        mask = spectrogram.ne(pad_value).all(dim=2)
        spectrogram_lengths = mask.sum(dim=1)
        waveform_lengths = self._get_waveform_lengths(spectrogram_lengths)

        spectrogram = spectrogram.transpose(2, 1)

        waveform = self.vocoder(spectrogram)

        outputs = []
        for output, length in zip(waveform, waveform_lengths):
            outputs.append(output[:length].unsqueeze(0))

        return outputs
