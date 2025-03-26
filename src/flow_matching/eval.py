import sys
import warnings
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoModelForTextToWaveform, AutoProcessor, pipeline

from .configs import ConditionalFlowMatchingWithBigVGanConfig, ConditionalFlowMatchingWithHifiGanConfig
from .data import UnitDataset
from .models import ConditionalFlowMatchingWithBigVGan, ConditionalFlowMatchingWithHifiGan
from .utils.misc import cer_transform, wer_transform

sys.path.append("src/utmos")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from ..utmos.score import Score

# register ConditionalFlowMatchingWithBigVGan
AutoConfig.register("flow_matching_with_bigvgan", ConditionalFlowMatchingWithBigVGanConfig)
AutoModelForTextToWaveform.register(ConditionalFlowMatchingWithBigVGanConfig, ConditionalFlowMatchingWithBigVGan)

# register ConditionalFlowMatchingWithHifiGan
AutoConfig.register("flow_matching_with_hifigan", ConditionalFlowMatchingWithHifiGanConfig)
AutoModelForTextToWaveform.register(ConditionalFlowMatchingWithHifiGanConfig, ConditionalFlowMatchingWithHifiGan)


@torch.inference_mode()
def evaluate(config):
    dataset = UnitDataset(config.dataset.test_file, config.dataset.wav_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        config.flow_matching_with_vocoder.batch_size,
        collate_fn=UnitDataset.collate_fn,
    )

    decoder = AutoModelForTextToWaveform.from_pretrained(config.flow_matching_with_vocoder.name).cuda()

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.asr.name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(config.asr.name)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=asr,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
    )

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

            hyp_wavs.append(hyp_wav.cpu().squeeze(0).numpy())
            ref_wavs.append(ref_wav.cpu().squeeze(0).numpy())
            hyp_scores.append(hyp_score)
            ref_scores.append(ref_score)

        batch_hyps = pipe(hyp_wavs, generate_kwargs={"language": "english"}, return_timestamps=True)
        batch_refs = pipe(ref_wavs, generate_kwargs={"language": "english"}, return_timestamps=True)

        transcripts += batch["transcripts"]
        hyps += [hyp["text"] for hyp in batch_hyps]
        refs += [ref["text"] for ref in batch_refs]

    wer_hyp = jiwer.wer(transcripts, hyps, wer_transform, wer_transform)
    cer_hyp = jiwer.cer(transcripts, hyps, cer_transform, cer_transform)
    mos_hyp = np.mean(hyp_scores)

    wer_ref = jiwer.wer(transcripts, refs, wer_transform, wer_transform)
    cer_ref = jiwer.cer(transcripts, refs, cer_transform, cer_transform)
    mos_ref = np.mean(ref_scores)

    Path(config.eval.result_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [wer_hyp, cer_hyp, mos_hyp, wer_ref, cer_ref, mos_ref],
        index=["WER (hyp)", "CER (hyp)", "MOS (hyp)", "WER (ref)", "CER (ref)", "MOS (ref)"],
    ).to_csv(config.eval.result_path)
