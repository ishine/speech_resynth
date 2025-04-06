import sys
import warnings
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import UnitDataset
from .models import ConditionalFlowMatchingWithBigVGan, ConditionalFlowMatchingWithHifiGan
from .utils.phi.normalizer import EnglishTextNormalizer
from .utils.phi.run_eval import Phi4MultimodalAudioModel

sys.path.append("src/utmos")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from ..utmos.score import Score


@torch.inference_mode()
def evaluate(config):
    dataset = UnitDataset(config.dataset.test_file, config.dataset.wav_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        config.flow_matching_with_vocoder.batch_size,
        collate_fn=UnitDataset.collate_fn,
    )

    decoder = ConditionalFlowMatchingWithBigVGan.load_pretrained(config.flow_matching.path, config.vocoder.path).cuda()

    asr = Phi4MultimodalAudioModel(config.asr.name)
    normalizer = EnglishTextNormalizer()

    scorer = Score(ckpt_path="src/utmos/epoch=3-step=7459.ckpt", input_sample_rate=16000, device="cuda")

    transcripts = []
    hyps = []
    refs = []
    hyp_scores = []
    ref_scores = []

    for batch in tqdm(dataloader):
        audio_values = decoder(
            batch["input_ids"].cuda(),
            config.flow_matching.dt,
            config.flow_matching.truncation_value,
        )

        hyp_wavs = []
        ref_wavs = []

        for hyp_wav, ref_wav in zip(audio_values, batch["input_values"]):
            hyp_score = scorer.score(hyp_wav.cuda())
            ref_score = scorer.score(ref_wav.cuda())

            hyp_wav = hyp_wav.cpu().squeeze(0).numpy()
            ref_wav = ref_wav.cpu().squeeze(0).numpy()

            hyp_wavs.append((hyp_wav, 16000))
            ref_wavs.append((ref_wav, 16000))
            hyp_scores.append(hyp_score)
            ref_scores.append(ref_score)

        batch_hyps = asr(hyp_wavs)
        batch_refs = asr(ref_wavs)

        transcripts += [normalizer(transcript) for transcript in batch["transcripts"]]
        hyps += [normalizer(hyp) for hyp in batch_hyps]
        refs += [normalizer(ref) for ref in batch_refs]

    wer_hyp = jiwer.wer(transcripts, hyps)
    cer_hyp = jiwer.cer(transcripts, hyps)
    mos_hyp = np.mean(hyp_scores)

    wer_ref = jiwer.wer(transcripts, refs)
    cer_ref = jiwer.cer(transcripts, refs)
    mos_ref = np.mean(ref_scores)

    Path(config.eval.result_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [wer_hyp, cer_hyp, mos_hyp, wer_ref, cer_ref, mos_ref],
        index=["WER (hyp)", "CER (hyp)", "MOS (hyp)", "WER (ref)", "CER (ref)", "MOS (ref)"],
    ).to_csv(config.eval.result_path)
