import random
from typing import Any, Dict, Optional

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths):
        self.wav_paths = list(wav_paths)

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        wav_path = self.wav_paths[n]
        name = wav_path.stem
        wav_path = str(wav_path)
        input_values, sr = torchaudio.load(wav_path)
        input_values = input_values.squeeze(0)
        return {"input_values": input_values, "name": name}

    @staticmethod
    def collate_fn(batch):
        input_values = [item["input_values"] for item in batch]
        attention_mask = [torch.ones_like(item["input_values"], dtype=torch.long) for item in batch]
        names = [item["name"] for item in batch]

        input_values = pad_sequence(input_values, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        wavs_len = torch.tensor([len(item["input_values"]) for item in batch])

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "wavs_len": wavs_len,
            "padding_mask": ~attention_mask.bool(),
            "names": names,
        }


def get_collate_fn(
    num_special_tokens: int = 2,
    pad_token_id: int = 0,
    units_per_sample: Optional[int] = None,
):
    def collate_fn(batch) -> Dict[str, torch.LongTensor]:
        input_ids = []
        names = []

        for item in batch:
            units = torch.tensor(item["units"]) + num_special_tokens

            if units_per_sample:
                diff = len(units) - units_per_sample

                if diff > 0:
                    start = random.randrange(diff)
                    units = units[start : start + units_per_sample]

            input_ids.append(units)
            names.append(item["id"])

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask = input_ids.ne(pad_token_id).long()
        labels = input_ids.masked_fill(input_ids.eq(pad_token_id), -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "names": names,
        }

    return collate_fn
