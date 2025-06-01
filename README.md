# Speech Resynthesis and Language Modeling with Flow Matching and Llama

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![model](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/ryota-komatsu/flow_matching_with_bigvgan)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-blue)](https://huggingface.co/datasets/ryota-komatsu/LibriTTS-R-whisper-large-v3-4096units)

## Setup

```shell
sudo apt install git-lfs  # for UTMOS

conda create -y -n py310 -c pytorch -c nvidia -c conda-forge python=3.10.17 pip=24.0 faiss-gpu=1.10.0
conda activate py310
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # optional

sh scripts/setup.sh  # download UTMOS
```

## Usage: sampling multi-speaker speech from supervised discrete units

```python
import torchaudio

from src.flow_matching.models import ConditionalFlowMatchingWithBigVGan
from src.flow_matching.utils.whisper import WhisperEncoder, WhisperFeatureExtractor

wav_path = "/path/to/wav"

# load model and processor
feature_extractor = WhisperFeatureExtractor.from_pretrained("ryota-komatsu/whisper-large-v3-tokenizer")
encoder = WhisperEncoder.from_pretrained("ryota-komatsu/whisper-large-v3-tokenizer").cuda()

# download a pretrained model from hugging face hub
decoder = ConditionalFlowMatchingWithBigVGan.from_pretrained("ryota-komatsu/flow_matching_with_bigvgan").cuda()

# load a waveform
waveform, sr = torchaudio.load(wav_path)
waveform = torchaudio.functional.resample(waveform, sr, 16000)

input_features = feature_extractor(
    waveform.squeeze(0).numpy(),
    return_tensors="pt",
    sampling_rate=16000,
    device="cuda",
    padding="do_not_pad",
).input_features.to("cuda")

# encode a waveform into pseudo-phonetic units
units = encoder.encode(input_features)
units = units.unsqueeze(0) + 1  # 0: pad

# resynthesis
audio_values = decoder(units)
```

## Usage: speech language modeling on subword units

```python
import torch
import torchaudio
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM

from src.flow_matching.utils.whisper import WhisperEncoder, WhisperFeatureExtractor
from src.speechlm.utils import convert_units_to_unicode

wav_path = "/path/to/wav"

# load model and processor
feature_extractor = WhisperFeatureExtractor.from_pretrained("ryota-komatsu/whisper-large-v3-tokenizer")
encoder = WhisperEncoder.from_pretrained("ryota-komatsu/whisper-large-v3-tokenizer").cuda()

# BPE tokenizer
tokenizer = Tokenizer.from_file("/path/to/pretrained/tokenizer.json")

model = LlamaForCausalLM.from_pretrained("/path/to/pretrained/model").cuda()

# load a waveform
waveform, sr = torchaudio.load(wav_path)
waveform = torchaudio.functional.resample(waveform, sr, 16000)

input_features = feature_extractor(
    waveform.squeeze(0).numpy(),
    return_tensors="pt",
    sampling_rate=16000,
    device="cuda",
    padding="do_not_pad",
).input_features.to("cuda")

# encode a waveform into pseudo-phonetic units
units = encoder.encode(input_features).tolist()
unicodes = convert_units_to_unicode(units)

# BPE
input_ids = tokenizer.encode(unicodes).ids
input_ids = torch.tensor([input_ids], device="cuda") + 2  # 0: pad, 1: EOS

# Speech LM
logits = model(input_ids=input_ids).logits
```

## Demo

Visit [demo page](https://ryota-komatsu.github.io/speech_resynth) for speech samples.

Jupyter notebook demo is found [here](demo.ipynb).

## Data Preparation

If you already have LibriTTS-R, you can use it by editing [a config file](configs/unit2speech/whisper-large-v3-4096-bigvgan.yaml#L7);
```yaml
dataset:
  wav_dir_orig: "/path/to/LibriTTS-R" # ${dataset.wav_dir_orig}/train-clean-100, train-clean-360, ...
```

otherwise you can download the new one under `dataset_root`.
```shell
dataset_root=data

sh scripts/download_libritts.sh ${dataset_root}
```

To perform speech language modeling, please download the Libri-Light under `dataset_root`.
```shell
dataset_root=data

sh scripts/download_librilight.sh ${dataset_root}  # 7TB
sh scripts/download_slm21.sh  # download sWUGGY and sBLIMP
```

## Training a unit-to-speech synthesizer

To run only a specific stage, pass it as an argument.

Supported processing stages
1. resample
1. extract_features  # can be skipped when using a pretrained BigVGan
1. train_bigvgan  # can be skipped when using a pretrained BigVGan
1. train_tokenizer  # can be skipped when using a pretrained model
1. tokenize_dataset  # can be skipped when using a Hugging Face datasets
1. train_flow_matching
1. synthesize

```shell
python main_resynth.py train_flow_matching --config=configs/unit2speech/whisper-large-v3-4096-bigvgan.yaml
```

## Training a speech language model

Set the number of GPUs to `nproc_per_node` to enable multi-GPU training.

```shell
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29400 \
    main_speechlm.py \
    --config=configs/speechlm/whisper.yaml
```

To run only a sub-task (encode, tokenize, or train), specify it as an argument.

```shell
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29400 \
    main_speechlm.py encode \
    --config=configs/speechlm/whisper.yaml
```

## Evaluation of a speech language model

See [Zero Resource Speech homepage](https://zerospeech.com/tasks/task_4/tasks_goals/) and [paper](https://arxiv.org/abs/2011.11588) for task details.

```shell
python main_speechlm.py eval --config=configs/speechlm/whisper.yaml
```