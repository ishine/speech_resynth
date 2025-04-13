import json
import random
from itertools import islice
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def fix_random_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_named_units_from_json(file, batch_size: int, num_special_tokens: int = 2) -> Dict[str, Any]:
    with open(file) as f:
        dataset = json.load(f)

    dataloader = iter(dataset.items())

    while True:
        batch = dict(islice(dataloader, batch_size))

        if not batch:
            break

        names = list(batch.keys())
        input_ids = [torch.tensor(value) + num_special_tokens for value in batch.values()]  # 0: pad, 1: eos
        input_ids = pad_sequence(input_ids, batch_first=True)

        yield {"names": names, "input_ids": input_ids}


def shift_unit(unit: int) -> int:
    """
    Avoid C0 control characters (0-31) and space (32)
    and map 0-93 to the printable ASCII range 33-126

    Avoid DEL (127), C1 control characters (128-159), and no-break space (160)
    and map 94- to 161-
    """
    if unit < 94:
        return unit + 33
    else:
        return unit + 67


def convert_units_to_unicode(units: Sequence[int]) -> str:
    """
    convert a unit sequence to a printable unicode string
    """
    return "".join(chr(shift_unit(u)) for u in units)


def get_lr_schedule(
    optimizer,
    total_steps: int,
    warmup_steps: int = 5000,
    base_lr: float = 1e-3,
    min_lr: float = 1e-4,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_schedule(current_step: int) -> float:
        if current_step < warmup_steps:
            return (min_lr + (base_lr - min_lr) * current_step / warmup_steps) / base_lr
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return (min_lr + (base_lr - min_lr) * (1 - progress)) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)


def get_num_non_embed_params(model):
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad and "embed_tokens" not in name and "lm_head" not in name:
            total += param.numel()
    return total / 1e6  # million
