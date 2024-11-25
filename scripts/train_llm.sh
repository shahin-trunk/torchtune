#!/bin/bash

#SBATCH --job-name=llm_ft_v7             # Job name
#SBATCH --nodes=2                        # Number of nodes
#SBATCH --ntasks-per-node=8              # Tasks per node (1 task per GPU)
#SBATCH --cpus-per-task=14               # CPUs per task
#SBATCH --gpus-per-node=8                # GPUs per node
#SBATCH --error=%x-%j.err                # Error log file
#SBATCH --output=%x-%j.out               # Output log file
#SBATCH --partition=nlp                  # Partition

# Pyxis and Enroot configurations
#SBATCH --container-image=/project/audio/workspce/audio/asr/nemo/image/sagi_v9.sqsh
#SBATCH --container-mounts=/project/audio/workspce/audio/asr/nemo:/workspace,/project:/project

# Set environment variables for PyTorch distributed
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Launch the training script inside the container
srun torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID /workspace/torchtune/recipes/lora_finetune_distributed.py --config ./llama_3_1_8B_lora_v9.yaml
