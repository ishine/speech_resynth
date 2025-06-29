from pathlib import Path

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .data import SpeechDataset
from .models import ConditionalFlowMatchingWithBigVGan
from .utils.textless import load_encoder


@torch.inference_mode()
def synthesize(config):
    dataset = SpeechDataset(
        config.synthesis.src_dir,
        split=config.synthesis.split,
        ext_audio=config.synthesis.ext_audio,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        config.flow_matching_with_vocoder.batch_size,
        collate_fn=SpeechDataset.collate_fn,
    )

    encoder = load_encoder(
        config.tokenizer.dense_model_name,
        config.tokenizer.quantizer_model_name,
        config.tokenizer.vocab_size,
        config.tokenizer.deduplicate,
    )

    decoder = ConditionalFlowMatchingWithBigVGan.load_pretrained(config.flow_matching.path, config.vocoder.path).cuda()

    for batch in tqdm(dataloader):
        input_ids = []
        for input_values, attention_mask in zip(batch["input_values"], batch["attention_mask"]):
            units = encoder(input_values[attention_mask.bool()].cuda())["units"]
            units = units + 1  # 0: pad
            input_ids.append(units)

        input_ids = pad_sequence(input_ids, batch_first=True)

        audio_values = decoder(input_ids, config.flow_matching.dt, config.flow_matching.truncation_value)

        for name, hyp_wav in zip(batch["names"], audio_values):
            hyp_wav = hyp_wav.cpu()

            hyp_path = Path(config.synthesis.tgt_dir) / name
            hyp_path = hyp_path.with_suffix(config.synthesis.ext_audio)
            hyp_path.parent.mkdir(parents=True, exist_ok=True)
            hyp_path = str(hyp_path)

            torchaudio.save(hyp_path, hyp_wav, 16000)
