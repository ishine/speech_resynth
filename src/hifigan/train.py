import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig

from .data import MelDataset, mel_spectrogram
from .models import MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss, feature_loss, generator_loss
from .utils import load_checkpoint, plot_spectrogram, save_checkpoint

torch.backends.cudnn.benchmark = True


def train(rank, config):
    if config.hifigan.num_gpus > 1:
        init_process_group(
            backend=config.hifigan.dist_config.dist_backend,
            init_method=config.hifigan.dist_config.dist_url,
            world_size=config.hifigan.dist_config.world_size * config.hifigan.num_gpus,
            rank=rank,
        )

    torch.cuda.manual_seed(config.hifigan.seed)
    device = torch.device(f"cuda:{rank:d}")

    generator = FastSpeech2ConformerHifiGan(
        FastSpeech2ConformerHifiGanConfig(
            upsample_rates=list(config.hifigan.upsample_rates),
            upsample_kernel_sizes=list(config.hifigan.upsample_kernel_sizes),
            normalize_before=False,
        )
    ).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    generator.apply_weight_norm()

    if rank == 0:
        print(generator)
        os.makedirs(config.hifigan.path, exist_ok=True)
        print("checkpoints directory : ", config.hifigan.path)

    cp_do = os.path.join(config.hifigan.path, "do") if os.path.isfile(os.path.join(config.hifigan.path, "do")) else None

    steps = 0
    if cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        generator = FastSpeech2ConformerHifiGan.from_pretrained(config.hifigan.path).to(device)
        state_dict_do = load_checkpoint(cp_do, device)
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    if config.hifigan.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(), config.hifigan.learning_rate, betas=[config.hifigan.adam_b1, config.hifigan.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        config.hifigan.learning_rate,
        betas=[config.hifigan.adam_b1, config.hifigan.adam_b2],
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.hifigan.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.hifigan.lr_decay)

    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])
        scheduler_g.load_state_dict(state_dict_do["scheduler_g"])
        scheduler_d.load_state_dict(state_dict_do["scheduler_d"])
        scaler_g.load_state_dict(state_dict_do["scaler_g"])
        scaler_d.load_state_dict(state_dict_do["scaler_d"])

    trainset = MelDataset(
        config.dataset.wav_dir,
        config.dataset.spectrogram_dir,
        config.dataset.train_file,
        config.hifigan.segment_size,
        config.hifigan.n_fft,
        config.hifigan.hop_size,
        True,
        config.dataset.ext_audio,
    )

    train_sampler = DistributedSampler(trainset) if config.hifigan.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=config.hifigan.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        batch_size=config.hifigan.batch_size,
        pin_memory=True,
    )

    if rank == 0:
        validset = MelDataset(
            config.dataset.wav_dir,
            config.dataset.spectrogram_dir,
            config.dataset.dev_file,
            config.hifigan.segment_size,
            config.hifigan.n_fft,
            config.hifigan.hop_size,
            False,
            config.dataset.ext_audio,
        )
        validation_loader = DataLoader(validset, num_workers=config.hifigan.num_workers, pin_memory=True)

        sw = SummaryWriter(os.path.join(config.hifigan.path, "logs"))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), config.hifigan.training_epochs):
        if rank == 0:
            start = time.time()
            print(f"Epoch: {epoch + 1}")

        if config.hifigan.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            with torch.amp.autocast("cuda"):
                x, y, mask = batch
                x = x.to(device)
                y = y.to(device)
                y = y.unsqueeze(1)

                y_g_hat = generator(x.transpose(1, 2)).unsqueeze(1)
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1))

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f

            optim_d.zero_grad()
            scaler_d.scale(loss_disc_all).backward()
            scaler_d.step(optim_d)
            scaler_d.update()

            # Generator
            with torch.amp.autocast("cuda"):
                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(x[mask], y_g_hat_mel[mask]) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            optim_g.zero_grad()
            scaler_g.scale(loss_gen_all).backward()
            scaler_g.step(optim_g)
            scaler_g.update()

            if rank == 0:
                # STDOUT logging
                if steps % config.hifigan.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(x[mask], y_g_hat_mel[mask]).item()

                    print(
                        f"Steps : {steps:d}, Gen Loss Total : {loss_gen_all:4.3f}, Mel-Spec. Error : {mel_error:4.3f}, s/b : {time.time() - start_b:4.3f}",
                        flush=True,
                    )

                # checkpointing
                if steps % config.hifigan.checkpoint_interval == 0 and steps != 0:
                    if steps == config.vocoder.total_steps:
                        (generator.module if config.vocoder.num_gpus > 1 else generator).remove_weight_norm()

                    (generator.module if config.hifigan.num_gpus > 1 else generator).save_pretrained(
                        config.hifigan.path
                    )
                    save_checkpoint(
                        os.path.join(config.hifigan.path, "do"),
                        {
                            "mpd": (mpd.module if config.hifigan.num_gpus > 1 else mpd).state_dict(),
                            "msd": (msd.module if config.hifigan.num_gpus > 1 else msd).state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "optim_d": optim_d.state_dict(),
                            "scheduler_g": scheduler_g.state_dict(),
                            "scheduler_d": scheduler_d.state_dict(),
                            "scaler_g": scaler_g.state_dict(),
                            "scaler_d": scaler_d.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )
                    if steps == config.vocoder.total_steps:
                        return

                # Tensorboard summary logging
                if steps % config.hifigan.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % config.hifigan.validation_interval == 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.inference_mode():
                        for j, batch in enumerate(validation_loader):
                            x, y, mask = batch
                            x = x.to(device)
                            y_g_hat = generator(x.transpose(1, 2)).unsqueeze(1)
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1))
                            val_err_tot += F.l1_loss(x[mask], y_g_hat_mel[mask]).item()

                            if j < 5:
                                if steps == 0:
                                    sw.add_audio(f"gt/y_{j}", y[0], steps, 16000)
                                    sw.add_figure(f"gt/y_spec_{j}", plot_spectrogram(x[0].cpu()), steps)

                                sw.add_audio(f"generated/y_hat_{j}", y_g_hat[0], steps, 16000)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1))
                                sw.add_figure(
                                    f"generated/y_hat_spec_{j}",
                                    plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()),
                                    steps,
                                )

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")


def train_hifigan(config):
    torch.manual_seed(config.hifigan.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.hifigan.seed)
        config.hifigan.num_gpus = torch.cuda.device_count()
        config.hifigan.batch_size = int(config.hifigan.batch_size / config.hifigan.num_gpus)
        print("Batch size per GPU :", config.hifigan.batch_size)
    else:
        pass

    if config.hifigan.num_gpus > 1:
        mp.spawn(
            train,
            nprocs=config.hifigan.num_gpus,
            args=(config,),
        )
    else:
        train(0, config)
