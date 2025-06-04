from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from textless.data.speech_encoder import SpeechEncoder
from tqdm import tqdm

from .data import SpeechDataset


def _tokenize(
    encoder: SpeechEncoder,
    data_loader: torch.utils.data.DataLoader,
):
    features = Features(
        {
            "id": Value("string"),
            "units": Sequence(Value("int32")),
        }
    )

    def generate_dataset():
        for item in tqdm(data_loader):
            outputs = encoder(item["input_values"].cuda())
            units = outputs["units"].tolist()

            yield {"id": item["name"][0], "units": units}

    return Dataset.from_generator(generate_dataset, features=features)


def tokenize_slm21(config):
    app_dir = Path(config.dataset.APP_DIR).expanduser()

    swuggy_dir = app_dir / "datasets/sLM21-dataset/lexical"
    sblimp_dir = app_dir / "datasets/sLM21-dataset/syntactic"

    swuggy_dev_paths = list(swuggy_dir.glob("dev/*.wav"))
    sblimp_dev_paths = list(sblimp_dir.glob("dev/*.wav"))
    swuggy_test_paths = list(swuggy_dir.glob("test/*.wav"))
    sblimp_test_paths = list(sblimp_dir.glob("test/*.wav"))

    swuggy_dev_set = SpeechDataset(swuggy_dev_paths)
    sblimp_dev_set = SpeechDataset(sblimp_dev_paths)
    swuggy_test_set = SpeechDataset(swuggy_test_paths)
    sblimp_test_set = SpeechDataset(sblimp_test_paths)

    swuggy_dev_loader = torch.utils.data.DataLoader(swuggy_dev_set)
    sblimp_dev_loader = torch.utils.data.DataLoader(sblimp_dev_set)
    swuggy_test_loader = torch.utils.data.DataLoader(swuggy_test_set)
    sblimp_test_loader = torch.utils.data.DataLoader(sblimp_test_set)

    encoder = SpeechEncoder.by_name(
        dense_model_name=config.s2u.dense_model_name,
        quantizer_model_name=config.s2u.quantizer_model_name,
        vocab_size=config.s2u.vocab_size,
        deduplicate=True,
        need_f0=False,
    ).cuda()

    swuggy_dev = _tokenize(encoder, swuggy_dev_loader)
    sblimp_dev = _tokenize(encoder, sblimp_dev_loader)
    swuggy_test = _tokenize(encoder, swuggy_test_loader)
    sblimp_test = _tokenize(encoder, sblimp_test_loader)

    swuggy = DatasetDict({"dev": swuggy_dev, "test": swuggy_test})
    sblimp = DatasetDict({"dev": sblimp_dev, "test": sblimp_test})

    swuggy.push_to_hub(config.dataset.swuggy)
    sblimp.push_to_hub(config.dataset.sblimp)


def tokenize_trainset(config, spk_ids: str = "1-9"):
    wav_dir_train = Path(config.dataset.wav_dir_train)
    train_paths = wav_dir_train.glob(f"*/[{spk_ids}]*/**/*" + config.dataset.ext_audio)
    train_set = SpeechDataset(train_paths)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=config.s2u.num_workers)

    encoder = SpeechEncoder.by_name(
        dense_model_name=config.s2u.dense_model_name,
        quantizer_model_name=config.s2u.quantizer_model_name,
        vocab_size=config.s2u.vocab_size,
        deduplicate=True,
        need_f0=False,
    ).cuda()

    trainset = _tokenize(encoder, train_loader)
    trainset.push_to_hub(config.dataset.train, split=f"train{spk_ids}")
