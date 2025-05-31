from pathlib import Path

import librosa
import torch
import torchaudio
from tqdm import tqdm


def resample(config):
    wav_dir_orig = Path(config.dataset.wav_dir_orig)
    wav_dir = Path(config.dataset.wav_dir)
    wav_paths = list(wav_dir_orig.glob("**/*" + config.dataset.ext_audio))

    for wav_path in tqdm(wav_paths):
        wav_name = wav_path.relative_to(wav_dir_orig)
        wav_path = str(wav_path)

        wav, sr = torchaudio.load(wav_path)
        wav = torchaudio.functional.resample(wav, sr, 16000)

        if config.dataset.vad:
            wav = wav.numpy()
            wav, _ = librosa.effects.trim(wav, top_db=20)
            wav = torch.from_numpy(wav)

        wav_path = wav_dir / wav_name
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path = str(wav_path)  # for sox backend
        torchaudio.save(wav_path, wav, 16000)
