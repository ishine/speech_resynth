# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

import math

import faiss
import numpy as np
import torch
import transformers
from datasets import Array2D, Dataset, DatasetDict
from torch import nn
from tqdm import tqdm
from transformers import WhisperConfig
from transformers.audio_utils import spectrogram, window_function
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperDecoder, WhisperEncoderLayer, WhisperPreTrainedModel

from ...bigvgan.data import mel_spectrogram
from ..data import LibriTTS_R


class WhisperFeatureExtractor(transformers.WhisperFeatureExtractor):
    def _np_extract_fbank_features(self, waveform_batch: np.array, device: str) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
        if device != "cpu":
            raise ValueError(
                f"Got device `{device}` for feature extraction, but feature extraction on CUDA accelerator "
                "devices requires torch, which is not installed. Either set `device='cpu'`, or "
                "install torch according to the official instructions: https://pytorch.org/get-started/locally/"
            )
        log_spec_batch = []
        for waveform in waveform_batch:
            log_spec = spectrogram(
                waveform,
                window_function(self.n_fft, "hann"),
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                dither=self.dither,
                mel_filters=self.mel_filters,
                log_mel="log10",
            )
            # log_spec = log_spec[:, :-1]
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            log_spec_batch.append(log_spec)
        log_spec_batch = np.array(log_spec_batch)
        return log_spec_batch

    def _torch_extract_fbank_features(self, waveform: np.array, device: str = "cpu") -> np.ndarray:
        """
        Compute the log-mel spectrogram of the audio using PyTorch's GPU-accelerated STFT implementation with batching,
        yielding results similar to cpu computing with 1e-5 tolerance.
        """
        waveform = torch.from_numpy(waveform).to(device, torch.float32)
        window = torch.hann_window(self.n_fft, device=device)

        # Note: it would be better to dither the chunked waveform,
        # so overlapping signal does not get the same dithering.
        # But, chunking is happening inside pytorch, so it is here.
        if self.dither != 0.0:
            waveform += self.dither * torch.randn(waveform.shape, dtype=waveform.dtype, device=waveform.device)

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft.abs() ** 2  # magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        if device != "cpu":
            log_spec = log_spec.detach().cpu()
        return log_spec.numpy()


class WhisperEncoder(WhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self.register_buffer("quantizer", torch.zeros(config.codebook_size, config.d_model))

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
    ):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos[: inputs_embeds.shape[1]]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                layer_head_mask=None,
                output_attentions=None,
            )

            hidden_states = layer_outputs[0]

        # hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)

    @torch.inference_mode()
    def encode(self, input_features, **kwargs):
        assert input_features.shape[0] == 1

        hidden_state = self(input_features, **kwargs).last_hidden_state
        hidden_state = hidden_state.squeeze(0)

        # K-means
        units = torch.cdist(hidden_state, self.quantizer).argmin(1)

        return units


class WhisperModel(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()


@torch.amp.autocast("cuda", dtype=torch.float16)
@torch.inference_mode()
def train_tokenizer(config):
    # load model and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.tokenizer.base)
    model_config = WhisperConfig.from_pretrained(config.tokenizer.base)
    model_config.codebook_size = config.tokenizer.vocab_size
    model = WhisperModel.from_pretrained(
        config.tokenizer.base, config=model_config, torch_dtype=torch.float16
    ).encoder.cuda()

    # load dataset
    dataset = LibriTTS_R(config.dataset.wav_dir, split="train-clean-100")
    loader = torch.utils.data.DataLoader(dataset)

    hidden_states = []

    for batch in tqdm(loader):
        input_features = feature_extractor(
            batch["input_values"].squeeze(0).numpy(),
            return_tensors="pt",
            sampling_rate=16000,
            device="cuda",
            padding="do_not_pad",
        ).input_features.to("cuda")
        length = model._get_feat_extract_output_lengths(input_features.shape[2])

        try:
            hidden_state = model(input_features).last_hidden_state

            hidden_state = hidden_state[0, :length].cpu().numpy()
            hidden_states.append(hidden_state)
        except:
            pass

    hidden_states = np.concatenate(hidden_states)

    quantizer = faiss.Kmeans(
        hidden_states.shape[1],
        config.tokenizer.vocab_size,
        niter=100,
        nredo=5,
        verbose=True,
        seed=0,
        gpu=True,
        min_points_per_centroid=1,
        max_points_per_centroid=hidden_states.shape[0],
    )
    quantizer.train(hidden_states)

    model.quantizer = torch.from_numpy(quantizer.centroids)
    model.push_to_hub(config.tokenizer.name)
    feature_extractor.push_to_hub(config.tokenizer.name)


def tokenize_dataset(config):
    # load model and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.tokenizer.name)
    model = WhisperEncoder.from_pretrained(config.tokenizer.name).cuda()

    trainset = LibriTTS_R(config.dataset.wav_dir, split="train-*")
    devset = LibriTTS_R(config.dataset.wav_dir, split="dev-clean")

    train_loader = torch.utils.data.DataLoader(trainset)
    dev_loader = torch.utils.data.DataLoader(devset)

    trainset = _encode(feature_extractor, model, train_loader)
    devset = _encode(feature_extractor, model, dev_loader)

    dataset = DatasetDict({"train": trainset, "dev": devset})
    dataset.push_to_hub(config.dataset.name)


def _encode(feature_extractor, model, dataloader: torch.utils.data.DataLoader):
    dataset = []

    for item in tqdm(dataloader):
        input_features = feature_extractor(
            item["input_values"].squeeze(0).numpy(),
            return_tensors="pt",
            sampling_rate=16000,
            device="cuda",
            padding="do_not_pad",
        ).input_features.to("cuda")

        input_values = item["input_values"].cuda()
        input_values = input_values / input_values.abs().max() * 0.95

        spectrogram_labels = mel_spectrogram(input_values, center=True).squeeze(0)  # (80, len)
        spectrogram_labels = spectrogram_labels.transpose(0, 1)  # (len, 80)
        spectrogram_labels = spectrogram_labels.cpu().tolist()

        try:
            units = model.encode(input_features).tolist()

            item = {
                "id": item["name"][0],
                "units": units,
                "transcript": item["transcript"][0],
                "spectrogram": spectrogram_labels,
            }
            dataset.append(item)
        except:
            pass

    dataset = Dataset.from_list(dataset)
    dataset = dataset.cast_column("spectrogram", Array2D(shape=(None, 80), dtype="float32"))
    return dataset
