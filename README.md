# Speech Resynthesis and Language Modeling with Flow Matching and Llama

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org)
[![model](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/ryota-komatsu/flow_matching_with_bigvgan)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-blue)](https://huggingface.co/datasets/ryota-komatsu/libritts-r-mhubert-2000units)
[![demo](https://img.shields.io/badge/Demo-blue)](https://ryota-komatsu.github.io/speech_resynth/)

## Setup

```shell
sudo apt install git-lfs  # for UTMOS

conda create -y -n py39 python=3.9.21 pip=24.0
conda activate py39
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # optional

sh scripts/setup.sh  # download textlesslib and UTMOS

cd src/textlesslib
pip install -e .
cd -
```

## Usage: sampling multi-speaker speech from self-supervised discrete units

```python
import torchaudio
from textless.data.speech_encoder import SpeechEncoder

from src.flow_matching.models import ConditionalFlowMatchingWithBigVGan

wav_path = "/path/to/wav"

encoder = SpeechEncoder.by_name(
    dense_model_name="mhubert-base-vp_mls_cv_8lang",
    quantizer_model_name="kmeans-expresso",
    vocab_size=2000,
    deduplicate=False,
    need_f0=False,
).cuda()

# download a pretrained model from hugging face hub
decoder = ConditionalFlowMatchingWithBigVGan.from_pretrained("ryota-komatsu/flow_matching_with_bigvgan").cuda()

# load a waveform
waveform, sr = torchaudio.load(wav_path)
waveform = torchaudio.functional.resample(waveform, sr, 16000)

# encode a waveform into pseudo-phonetic units
units = encoder(waveform.cuda())["units"]
units = units.unsqueeze(0) + 1  # 0: pad

# resynthesis
audio_values = decoder(units)
```

## Usage: speech language modeling on subword units

```python
import torch
import torchaudio
from textless.data.speech_encoder import SpeechEncoder
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM

from src.speechlm.utils import convert_units_to_unicode

wav_path = "/path/to/wav"

encoder = SpeechEncoder.by_name(
    dense_model_name="hubert-base-ls960",
    quantizer_model_name="kmeans",
    vocab_size=100,
    deduplicate=True,
    need_f0=False,
).cuda()

# BPE tokenizer
tokenizer = Tokenizer.from_file("/path/to/pretrained/tokenizer.json")

model = LlamaForCausalLM.from_pretrained("/path/to/pretrained/model").cuda()

# load a waveform
waveform, sr = torchaudio.load(wav_path)
waveform = torchaudio.functional.resample(waveform, sr, 16000)

# encode a waveform into pseudo-phonetic units
units = encoder(waveform.cuda())["units"].tolist()
unicodes = convert_units_to_unicode(units)

# BPE
input_ids = tokenizer.encode(unicodes).ids
input_ids = torch.tensor([input_ids], device="cuda") + 2  # 0: pad, 1: EOS

# Speech LM
logits = model(input_ids=input_ids).logits
```

## Demo

Visit [demo page](https://ryota-komatsu.github.io/speech_resynth) for speech samples.

## Data Preparation

If you already have LibriTTS-R, you can use it by editing [a config file](configs/unit2speech/mhubert-expresso-2000.yaml#L7);
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
1. tokenize_dataset  # can be skipped when using a Hugging Face datasets
1. train_flow_matching
1. synthesize

```shell
python main_resynth.py train_flow_matching --config=configs/unit2speech/mhubert-expresso-2000.yaml
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
    --config=configs/speechlm/hubert.yaml
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
    --config=configs/speechlm/hubert.yaml
```

## Evaluation of a speech language model

See [Zero Resource Speech homepage](https://zerospeech.com/tasks/task_4/tasks_goals/) and [paper](https://arxiv.org/abs/2011.11588) for task details.

```shell
python main_speechlm.py eval --config=configs/speechlm/hubert.yaml
```