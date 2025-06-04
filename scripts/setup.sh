#!/bin/sh

git clone https://github.com/facebookresearch/libri-light.git src/libri-light
git clone https://github.com/facebookresearch/textlesslib.git src/textlesslib
git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo src/utmos

patch src/utmos/lightning_module.py src/patch/utmos_lightning_module.patch