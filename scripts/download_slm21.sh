#!/bin/sh

dataset_root=${1:-data/zr-data/datasets}

wget -t 0 -c -P ${dataset_root} https://download.zerospeech.com/datasets/sLM21.dataset.zip

cd ${dataset_root}
unzip -d sLM21-dataset sLM21.dataset.zip
cd -