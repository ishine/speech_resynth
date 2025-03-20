#!/bin/bash

#$ -cwd                      ## Execute a job in the current directory
#$ -l node_h=1               ## Use number of node
#$ -l h_rt=24:00:00          ## Running job time
#$ -p -5
#$ -m abe
#$ -M EMAIL_ADDRESS

module load cuda/12.1.0
module load intel
module load cudnn/9.0.0
module load nccl/2.20.5

module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate py39

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29400 \
    main_speechlm.py train \
    --config=configs/speechlm/hubert.yaml