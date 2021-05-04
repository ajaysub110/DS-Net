#!/bin/bash
#SBATCH --job-name=supernet
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:3
#SBATCH --constraint=gpu_6gb|gpu_12gb
#SBATCH --mem=64G
#SBATCH -c 3

# activate conda env
source /misc/kcgscratch1/ajay_exp/miniconda3/etc/profile.d/conda.sh
conda activate dsnet

# run supernet training
python -u -m torch.distributed.launch --nproc_per_node=3 train.py ./data/ -c ./configs/mobilenetv1_bn_uniform.yml
