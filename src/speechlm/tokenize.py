import glob
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

from ..flow_matching.utils.whisper import WhisperEncoder, WhisperFeatureExtractor
from .data import SpeechDataset
from .utils import convert_units_to_unicode, shift_unit


def tokenize(config):
    Path(config.s2u.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)

    files = glob.glob(config.dataset.unicode_train + "*")
    initial_alphabet = [chr(shift_unit(unit)) for unit in range(config.s2u.vocab_size)]
    trainer = BpeTrainer(vocab_size=config.model.vocab_size, initial_alphabet=initial_alphabet)
    tokenizer = Tokenizer(BPE())
    tokenizer.train(files=files, trainer=trainer)
    tokenizer.save(config.s2u.tokenizer_path)

    Path(config.dataset.train_file).parent.mkdir(parents=True, exist_ok=True)
    with open(config.dataset.train_file, "w") as f:
        for file in files:
            with open(file) as g:
                for unicodes in g:
                    unicodes = unicodes.rstrip()
                    units = tokenizer.encode(unicodes).ids
                    units = " ".join(str(u) for u in units)

                    f.write(f"{units}\n")


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

    # load model and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.s2u.name)
    encoder = WhisperEncoder.from_pretrained(config.s2u.name).cuda()
    tokenizer = Tokenizer.from_file(config.s2u.tokenizer_path)

    _tokenize_slm21(feature_extractor, encoder, tokenizer, config.dataset.swuggy_dev_file, swuggy_dev_loader)
    _tokenize_slm21(feature_extractor, encoder, tokenizer, config.dataset.sblimp_dev_file, sblimp_dev_loader)
    _tokenize_slm21(feature_extractor, encoder, tokenizer, config.dataset.swuggy_test_file, swuggy_test_loader)
    _tokenize_slm21(feature_extractor, encoder, tokenizer, config.dataset.sblimp_test_file, sblimp_test_loader)


def _tokenize_slm21(
    feature_extractor: WhisperFeatureExtractor,
    encoder: WhisperEncoder,
    tokenizer: Tokenizer,
    file,
    data_loader: torch.utils.data.DataLoader,
):
    Path(file).parent.mkdir(parents=True, exist_ok=True)

    dataset = dict()

    for item in tqdm(data_loader):
        input_features = feature_extractor(
            item["input_values"].squeeze(0).numpy(),
            return_tensors="pt",
            sampling_rate=16000,
            device="cuda",
            padding="do_not_pad",
        ).input_features.to("cuda")
        units = encoder(input_features, out_layer=15).tolist()
        unicodes = convert_units_to_unicode(units)
        input_ids = tokenizer.encode(unicodes).ids

        dataset[item["name"][0]] = input_ids

    with open(file, "w") as f:
        json.dump(dataset, f)


def encode(config, spk_ids: str = "1-9"):
    wav_dir_train = Path(config.dataset.wav_dir_train)
    train_paths = wav_dir_train.glob(f"*/[{spk_ids}]*/**/*" + config.dataset.ext_audio)
    train_set = SpeechDataset(train_paths)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=config.s2u.num_workers)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.s2u.name)
    encoder = WhisperEncoder.from_pretrained(config.s2u.name).cuda()

    _encode(feature_extractor, encoder, config.dataset.unicode_train + f"{spk_ids}", train_loader)


def _encode(
    feature_extractor: WhisperFeatureExtractor,
    encoder: WhisperEncoder,
    file,
    data_loader: torch.utils.data.DataLoader,
):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        for item in tqdm(data_loader):
            input_features = feature_extractor(
                item["input_values"].squeeze(0).numpy(),
                return_tensors="pt",
                sampling_rate=16000,
                device="cuda",
                padding="do_not_pad",
            ).input_features.to("cuda")
            units = encoder(input_features, out_layer=15).tolist()

            unicodes = convert_units_to_unicode(units)

            f.write(f"{unicodes}\n")
