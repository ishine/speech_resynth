# Copied and modified from https://github.com/huggingface/open_asr_leaderboard/blob/main/phi/run_eval.py

from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, StoppingCriteria, StoppingCriteriaList
from transformers.utils import is_flash_attn_2_available

torch.set_float32_matmul_precision("high")


class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        """Initialize the multiple token batch stopping criteria.

        Args:
            stop_tokens: Stop-tokens.
            batch_size: Batch size.

        """

        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only gather the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to `stop_tokens`
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        # Mark the position where a stop token has been produced for each input in the batch,
        # but only if the corresponding entry is not already set
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)


class Phi4MultimodalAudioModel:
    def __init__(self, name_or_path="microsoft/Phi-4-multimodal-instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
            device_map="cuda",
        )
        self.processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True)

        user = "<|user|>"
        assistant = "<|assistant|>"
        self.prompt_suffix = "<|end|>"
        self.prompt = f"{user}<|audio_1|>Transcribe the audio clip into text.{self.prompt_suffix}{assistant}"

    def __call__(
        self,
        audios: List[Tuple[np.ndarray, int]],
        max_new_tokens: int = 512,
        num_beams: int = 1,
        num_logits_to_keep: int = 0,
    ) -> List[str]:
        """
        Args:
            audios (`List[Tuple[np.ndarray, int]]`):
                list of (1D array audio, sampling rate).
        Returns:
            pred_text (`List[str]`):
                batched unnormalized transcripts.
        """
        stop_tokens = [self.prompt_suffix, self.processor.tokenizer.eos_token]
        stop_tokens_ids = self.processor.tokenizer(
            stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt"
        )["input_ids"]
        stop_tokens_ids = stop_tokens_ids.to(self.model.device)

        # Load audio inputs
        minibatch_size = len(audios)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "stopping_criteria": StoppingCriteriaList(
                [MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=num_beams * minibatch_size)]
            ),
        }

        with torch.autocast(self.model.device.type, enabled=True):
            inputs = self.processor(text=[self.prompt] * minibatch_size, audios=audios, return_tensors="pt").to(
                self.model.device
            )

            # Model Inference
            pred_ids = self.model.generate(
                **inputs,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **gen_kwargs,
                num_logits_to_keep=num_logits_to_keep,
            )

        # Gather the sequence index of the stop token
        stop_tokens_idx = gen_kwargs["stopping_criteria"][0].stop_tokens_idx.reshape(minibatch_size, -1)[:, 0]

        # If a stop token was produced, we need to remove its length from the found index,
        # however there might be a chance that the stop token was not produced and the index
        # returned is the length of the generated sequence
        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            pred_ids.shape[-1],
        )

        # Convert token ids to text transcription
        pred_text = [
            self.processor.decode(
                _pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for _pred_ids, _stop_tokens_idx in zip(pred_ids, stop_tokens_idx)
        ]
        return pred_text
