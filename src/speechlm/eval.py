import subprocess
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM

from .utils import load_named_units_from_json


def evaluate(config):
    model = LlamaForCausalLM.from_pretrained(config.model.path).cuda()

    num_special_tokens = len(
        {
            token_id
            for token_id in (config.model.pad_token_id, config.model.bos_token_id, config.model.eos_token_id)
            if token_id is not None
        }
    )

    _eval(
        model,
        config.dataset.swuggy_test_file,
        Path(config.dataset.result_dir) / "lexical/test.txt",
        config.dataloader.batch_size_per_device,
        num_special_tokens,
    )
    _eval(
        model,
        config.dataset.sblimp_test_file,
        Path(config.dataset.result_dir) / "syntactic/test.txt",
        config.dataloader.batch_size_per_device,
        num_special_tokens,
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
    model: LlamaForCausalLM,
    in_file,
    out_file,
    batch_size: int,
    num_special_tokens: int = 2,
):
    with open(out_file, "w") as f:
        for batch in load_named_units_from_json(in_file, batch_size, num_special_tokens):
            # Speech LM
            input_ids = batch["input_ids"].cuda()
            labels = input_ids.masked_fill(input_ids.eq(0), -100)
            logits = model(input_ids=input_ids, labels=labels).logits.transpose(1, 2)

            labels = F.pad(labels, (0, 1), value=-100)
            shifted_labels = labels[:, 1:]

            scores = -F.cross_entropy(logits, shifted_labels, reduction="none")
            scores = scores.sum(dim=1) / scores.ne(0).sum(dim=1)
            scores = scores.tolist()

            for name, score in zip(batch["names"], scores):
                f.write(f"{name} {score}\n")
