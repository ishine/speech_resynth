import random
from typing import Any, Dict

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


class UnitDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        files,
        units_per_sample: int,
        num_special_tokens: int = 2,
        eos_token_id: int = 1,
    ):
        self.input_ids = []
        for file in files:
            with open(file) as f:
                for units in f:
                    units = units.rstrip().split()
                    units = torch.tensor([int(u) + num_special_tokens for u in units] + [eos_token_id])
                    self.input_ids.append(units)

        self.units_per_sample = units_per_sample

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, n: int) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids[n]
        attention_mask = torch.ones_like(input_ids)

        diff = len(input_ids) - self.units_per_sample

        if diff > 0:
            start = random.randrange(diff)
            input_ids = input_ids[start : start + self.units_per_sample]
            attention_mask = attention_mask[start : start + self.units_per_sample]
        else:
            input_ids = torch.nn.functional.pad(input_ids, (0, -diff))
            attention_mask = torch.nn.functional.pad(attention_mask, (0, -diff))

        labels = input_ids.masked_fill(input_ids.eq(0), -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
