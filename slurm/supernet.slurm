#!/bin/bash
#SBATCH --job-name=supernet
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:3
#SBATCH --constraint=gpu_6gb|gpu_12gb
#SBATCH --mem=64G
#SBATCH -c 3

# activate conda env
conda deactivate
conda deactivate
conda activate dsnet

# run supernet training
python -m torch.distributed.launch --nproc_per_node=3 train.py ./data/ -c ./configs/mobilenetv1_bn_uniform.yml
