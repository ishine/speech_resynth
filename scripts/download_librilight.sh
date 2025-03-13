#!/bin/sh

dataset_root=${1:-data}

wget -t 0 -c -P ${dataset_root}/librilight https://dl.fbaipublicfiles.com/librilight/data/small.tar
wget -t 0 -c -P ${dataset_root}/librilight https://dl.fbaipublicfiles.com/librilight/data/medium.tar
wget -t 0 -c -P ${dataset_root}/librilight https://dl.fbaipublicfiles.com/librilight/data/large.tar

tar xvf ${dataset_root}/librilight/small.tar -C ${dataset_root}/librilight
tar xvf ${dataset_root}/librilight/medium.tar -C ${dataset_root}/librilight
tar xvf ${dataset_root}/librilight/large.tar -C ${dataset_root}/librilight