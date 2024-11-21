#!/bin/bash

#SBATCH --job-name=llm_ft_v7
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --partition=nlp

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_USE_CUDA_DSA=1
export OMP_NUM_THREADS=12
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ENABLE_MONITORING=0
#export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=12800000
export NCCL_IB_TIMEOUT=36000
export NCCL_DEBUG=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export KMP_AFFINITY=disabled
export OMP_NUM_THREADS=8
export PYTHONFAULTHANDLER=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export NEMO_DATA_STORE_CACHE_DIR="/project/audio/data/asr/shared/cache"
export NEMO_DATA_STORE_CACHE_SHARED=1
#export NCCL_IB_DISABLE=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
#export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)

export SHARED_STORAGE_ROOT="/project/audio/workspce/audio/asr/nemo"
export CONTAINER_IMAGE="$SHARED_STORAGE_ROOT/image/sagi_v9.sqsh"
#export CONTAINER_IMAGE="nvcr.io/nvidia/nemo:24.01.speech"
export CONTAINER_MOUNTS="$SHARED_STORAGE_ROOT:/workspace,/project:/project"

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
#echo "slurm_nodes: $slurm_nodes"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "MASTER_ADDRESS: $MASTER_ADDRESS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

srun --container-mounts=$CONTAINER_MOUNTS\
     --container-image=$CONTAINER_IMAGE \
     --container-workdir /workspace \
     torchrun --nnodes 2 --nproc-per-node 8 --rdzv_endpoint=localhost:29400 /workspace/torchtune/recipes/lora_finetune_distributed.py --config ./llama_3_1_8B_lora_v7.yaml


