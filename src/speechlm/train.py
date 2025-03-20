import os
import subprocess
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import LlamaConfig, LlamaForCausalLM

from .data import UnitDataset
from .eval import _eval
from .utils import fix_random_seed, get_lr_schedule


@torch.inference_mode()
def validate(config, model, step: int, writer: SummaryWriter, num_special_tokens: int = 2):
    torch.cuda.empty_cache()
    model.eval()

    os.environ["APP_DIR"] = str(Path(config.dataset.APP_DIR).expanduser())

    if not Path(config.dataset.result_dir).is_dir():
        subprocess.run(["zrc", "submission:init", "sLM21", config.dataset.result_dir], env=os.environ)

    _eval(
        model,
        config.dataset.swuggy_dev_file,
        Path(config.dataset.result_dir) / "lexical/dev.txt",
        config.dataloader.batch_size_per_device,
        num_special_tokens,
    )
    _eval(
        model,
        config.dataset.sblimp_dev_file,
        Path(config.dataset.result_dir) / "syntactic/dev.txt",
        config.dataloader.batch_size_per_device,
        num_special_tokens,
    )

    subprocess.run(
        [
            "zrc",
            "benchmarks:run",
            "sLM21",
            config.dataset.result_dir,
            "--sets",
            "dev",
            "--task",
            "lexical",
            "syntactic",
        ]
    )

    df_swuggy = pd.read_csv(Path(config.dataset.result_dir) / "scores/score_lexical_dev_by_frequency.csv", index_col=0)
    df_sblimp = pd.read_csv(Path(config.dataset.result_dir) / "scores/score_syntactic_dev_by_type.csv", index_col=0)

    swuggy_all = (df_swuggy["n"] * df_swuggy["score"]).sum() / df_swuggy["n"].sum()
    swuggy_oov = df_swuggy.loc["oov", "score"]

    df_swuggy_iv = df_swuggy[df_swuggy.index != "oov"]
    swuggy_iv = (df_swuggy_iv["n"] * df_swuggy_iv["score"]).sum() / df_swuggy_iv["n"].sum()

    sblimp = (df_sblimp["n"] * df_sblimp["score"]).sum() / df_sblimp["n"].sum()

    writer.add_scalar("dev/sWUGGY all", swuggy_all, step)
    writer.add_scalar("dev/sWUGGY in-vocab", swuggy_iv, step)
    writer.add_scalar("dev/sWUGGY out-of-vocab", swuggy_oov, step)
    writer.add_scalar("dev/sBLIMP", sblimp, step)

    torch.cuda.empty_cache()


def train(config):
    fix_random_seed()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}.", flush=True)
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()

    num_special_tokens = len(
        {
            token_id
            for token_id in (config.model.pad_token_id, config.model.bos_token_id, config.model.eos_token_id)
            if token_id is not None
        }
    )

    trainset = UnitDataset(
        config.dataset.train_file,
        units_per_sample=config.dataset.units_per_sample,
        num_special_tokens=num_special_tokens,
        eos_token_id=config.model.eos_token_id,
    )
    sampler = DistributedSampler(trainset) if dist.is_initialized() else None
    train_loader = DataLoader(
        trainset,
        batch_size=config.dataloader.batch_size_per_device,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    if rank == 0:
        writer = SummaryWriter(config.model.path)

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=config.model.vocab_size + num_special_tokens,
            hidden_size=config.model.hidden_size,
            intermediate_size=config.model.intermediate_size,
            num_hidden_layers=config.model.num_hidden_layers,
            num_attention_heads=config.model.num_attention_heads,
            pad_token_id=config.model.pad_token_id,
            bos_token_id=config.model.bos_token_id,
            eos_token_id=config.model.eos_token_id,
        )
    ).to(device_id)
    model = DDP(model, device_ids=[device_id])

    optimizer = torch.optim.AdamW(model.parameters(), config.optim.lr, (config.optim.beta1, config.optim.beta2))

    # learning rate scheduler
    lr_scheduler = get_lr_schedule(
        optimizer,
        config.optim.total_steps,
        config.optim.warmup_steps,
        config.optim.lr,
        config.optim.lr_min,
    )

    scaler = torch.amp.GradScaler("cuda", init_scale=1e24)

    last_epoch = 0
    step = 0
    global_step = 0

    # resume training
    checkpoint_path = Path(config.model.path) / "checkpoint"
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, weights_only=True)

        last_epoch = ckpt["epoch"]
        step = ckpt["step"]
        global_step = ckpt["global_step"]
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])

        print(f"load from {checkpoint_path}", flush=True)
        del ckpt
        torch.cuda.empty_cache()

    for epoch in range(last_epoch, config.optim.epoch):
        model.train()

        if dist.is_initialized():
            sampler.set_epoch(epoch)

        train_loader_iter = iter(train_loader)

        for _ in range(step):
            next(train_loader_iter)

        for step, batch in enumerate(train_loader_iter, start=step):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    labels=batch["labels"].to(device_id),
                ).loss
            scaler.scale(loss).backward()

            # gradient clipping
            if config.optim.max_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_norm)

            # update model
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            optimizer.zero_grad()

            # update learning rate
            lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()

            step += 1
            global_step += 1

            # tensorboard log
            if rank == 0 and global_step % config.optim.summary_interval == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/scale", scale, global_step)
                if config.optim.max_norm is not None:
                    writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)

                # trace the peak GPU memory
                writer.add_scalar("memory/allocated (GB)", torch.cuda.max_memory_allocated() / 2**30, global_step)
                writer.add_scalar("memory/reserved (GB)", torch.cuda.max_memory_reserved() / 2**30, global_step)

            if rank == 0 and global_step % config.optim.validation_save_interval == 0:
                validate(config, model, global_step, writer, num_special_tokens)

                # save model
                ckpt = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                }
                Path(config.model.path).parent.mkdir(parents=True, exist_ok=True)
                model.module.save_pretrained(config.model.path)
                torch.save(ckpt, checkpoint_path)
                torch.save(ckpt, checkpoint_path.with_name(f"{checkpoint_path.name}{global_step:08}"))

                if global_step == config.optim.total_steps:
                    torch.distributed.destroy_process_group()
                    return

        step = 0

    torch.distributed.destroy_process_group()
