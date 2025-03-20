#!/bin/sh

dataset_root=${1:-data}
server=${2:-default}

if [ ${server} = "default" ]; then
    # default server
    wget -t 0 -c -P ${dataset_root} https://www.openslr.org/resources/141/train_clean_100.tar.gz
    wget -t 0 -c -P ${dataset_root} https://www.openslr.org/resources/141/train_clean_360.tar.gz
    wget -t 0 -c -P ${dataset_root} https://www.openslr.org/resources/141/train_other_500.tar.gz
    wget -t 0 -c -P ${dataset_root} https://www.openslr.org/resources/141/dev_clean.tar.gz
    wget -t 0 -c -P ${dataset_root} https://www.openslr.org/resources/141/dev_other.tar.gz
    wget -t 0 -c -P ${dataset_root} https://www.openslr.org/resources/141/test_clean.tar.gz
    wget -t 0 -c -P ${dataset_root} https://www.openslr.org/resources/141/test_other.tar.gz
elif [ ${server} = "cn" ]; then
    # CN mirror server
    wget -t 0 -c -P ${dataset_root} https://openslr.magicdatatech.com/resources/141/train_clean_100.tar.gz
    wget -t 0 -c -P ${dataset_root} https://openslr.magicdatatech.com/resources/141/train_clean_360.tar.gz
    wget -t 0 -c -P ${dataset_root} https://openslr.magicdatatech.com/resources/141/train_other_500.tar.gz
    wget -t 0 -c -P ${dataset_root} https://openslr.magicdatatech.com/resources/141/dev_clean.tar.gz
    wget -t 0 -c -P ${dataset_root} https://openslr.magicdatatech.com/resources/141/dev_other.tar.gz
    wget -t 0 -c -P ${dataset_root} https://openslr.magicdatatech.com/resources/141/test_clean.tar.gz
    wget -t 0 -c -P ${dataset_root} https://openslr.magicdatatech.com/resources/141/test_other.tar.gz
fi

tar zxvf ${dataset_root}/train_clean_100.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/train_clean_360.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/train_other_500.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/dev_clean.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/dev_other.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/test_clean.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/test_other.tar.gz -C ${dataset_root}