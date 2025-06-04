from pathlib import Path

import librosa
import torch
import torchaudio
from datasets import Array2D, Dataset, DatasetDict, Features, Sequence, Value
from tqdm import tqdm

from ..bigvgan.data import mel_spectrogram
from .data import LibriTTS_R
from .utils.textless import load_encoder


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


def tokenize_dataset(config):
    encoder = load_encoder(
        config.tokenizer.dense_model_name,
        config.tokenizer.quantizer_model_name,
        config.tokenizer.vocab_size,
        config.tokenizer.deduplicate,
    )

    trainset = LibriTTS_R(config.dataset.wav_dir, split="train-*")
    devset = LibriTTS_R(config.dataset.wav_dir, config.dataset.wav_dir_orig, split="dev-clean")

    train_loader = torch.utils.data.DataLoader(trainset)
    dev_loader = torch.utils.data.DataLoader(devset)

    trainset = _encode(encoder, train_loader)
    devset = _encode(encoder, dev_loader)

    dataset = DatasetDict({"train": trainset, "dev": devset})
    dataset.push_to_hub(config.dataset.name)


def _encode(encoder, dataloader: torch.utils.data.DataLoader):
    def generate_dataset():
        for item in tqdm(dataloader):
            input_values = item["input_values"].cuda()
            input_values = input_values / input_values.abs().max() * 0.95

            spectrogram_labels = mel_spectrogram(input_values).squeeze(0)  # (80, len)
            spectrogram_labels = spectrogram_labels.transpose(0, 1)  # (len, 80)
            spectrogram_labels = spectrogram_labels.cpu().tolist()

            try:
                units = encoder(item["input_values"].cuda())["units"].tolist()

                item = {
                    "id": item["name"][0],
                    "units": units,
                    "transcript": item["transcript"][0],
                    "spectrogram": spectrogram_labels,
                }
                yield item
            except:
                pass

    features = Features(
        {
            "id": Value("string"),
            "units": Sequence(Value("int32")),
            "transcript": Value("string"),
            "spectrogram": Array2D(shape=(None, 80), dtype="float32"),
        }
    )

    return Dataset.from_generator(generate_dataset, features=features)


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
