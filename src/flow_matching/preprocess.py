from pathlib import Path

import librosa
import torch
import torchaudio
from tqdm import tqdm

from ..bigvgan.data import mel_spectrogram


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


def extract_features(config):
    wav_dir = Path(config.dataset.wav_dir)
    spectrogram_dir = Path(config.dataset.spectrogram_dir)
    wav_paths = list(wav_dir.glob("**/*" + config.dataset.ext_audio))

    for wav_path in tqdm(wav_paths):
        wav_name = wav_path.relative_to(wav_dir).with_suffix("")
        spectrogram_path = spectrogram_dir / wav_name.with_suffix(".pt")
        if spectrogram_path.is_file():
            continue
        spectrogram_path.parent.mkdir(parents=True, exist_ok=True)

        wav_path = str(wav_path)
        wav, sr = torchaudio.load(wav_path)
        wav = wav.cuda()
        wav = wav / wav.abs().max() * 0.95

        spectrogram_labels = mel_spectrogram(wav)  # (1, 80, len)
        spectrogram_labels = spectrogram_labels.transpose(1, 2)  # (1, len, 80)
        spectrogram_labels = spectrogram_labels.cpu()

        torch.save(spectrogram_labels, spectrogram_path)
