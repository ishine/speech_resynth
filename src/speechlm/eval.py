import subprocess
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import get_collate_fn


def evaluate(config):
    model = AutoModelForCausalLM.from_pretrained(config.model.path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(config.model.path)

    _eval(
        model,
        tokenizer,
        config.dataset.swuggy,
        "test",
        Path(config.dataset.result_dir) / "lexical/test.txt",
        config.dataloader.batch_size_per_device,
    )
    _eval(
        model,
        tokenizer,
        config.dataset.sblimp,
        "test",
        Path(config.dataset.result_dir) / "syntactic/test.txt",
        config.dataloader.batch_size_per_device,
    )

    subprocess.run(
        [
            "zrc",
            "benchmarks:run",
            "sLM21",
            config.dataset.result_dir,
            "--skip-validation",
            "--sets",
            "test",
            "--task",
            "lexical",
            "syntactic",
        ]
    )

    df_swuggy = pd.read_csv(Path(config.dataset.result_dir) / "scores/score_lexical_test_by_frequency.csv", index_col=0)
    df_sblimp = pd.read_csv(Path(config.dataset.result_dir) / "scores/score_syntactic_test_by_type.csv", index_col=0)

    swuggy_all = (df_swuggy["n"] * df_swuggy["score"]).sum() / df_swuggy["n"].sum()
    swuggy_oov = df_swuggy.loc["oov", "score"]

    df_swuggy_iv = df_swuggy[df_swuggy.index != "oov"]
    swuggy_iv = (df_swuggy_iv["n"] * df_swuggy_iv["score"]).sum() / df_swuggy_iv["n"].sum()

    sblimp = (df_sblimp["n"] * df_sblimp["score"]).sum() / df_sblimp["n"].sum()

    pd.DataFrame(
        [swuggy_all, swuggy_iv, swuggy_oov, sblimp],
        index=["sWUGGY all", "sWUGGY in-vocab", "sWUGGY out-of-vocab", "sBLIMP"],
    ).to_csv(Path(config.dataset.result_dir) / "scores/score.csv")


@torch.inference_mode()
def _eval(
    model,
    tokenizer,
    in_file,
    split: str,
    out_file,
    batch_size: int,
):
    dataset = load_dataset(in_file, split=split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        collate_fn=get_collate_fn(tokenizer),
    )

    with open(out_file, "w") as f:
        for batch in loader:
            # Speech LM
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            logits = model(input_ids=input_ids, labels=labels).logits.transpose(1, 2)

            labels = F.pad(labels, (0, 1), value=-100)
            shifted_labels = labels[:, 1:]

            scores = -F.cross_entropy(logits, shifted_labels, reduction="none")
            scores = scores.sum(dim=1) / scores.ne(0).sum(dim=1)
            scores = scores.tolist()

            for name, score in zip(batch["names"], scores):
                f.write(f"{name} {score}\n")
