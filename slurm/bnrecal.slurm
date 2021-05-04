#!/bin/bash
#SBATCH --job-name=bnrecal
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --constraint=gpu_12gb
#SBATCH --mem=64G
#SBATCH -c 1

# activate conda env
source /misc/kcgscratch1/ajay_exp/miniconda3/etc/profile.d/conda.sh
conda activate dsnet

# run BN recalibration
nvidia-smi
python -u -m torch.distributed.launch --nproc_per_node=1 train.py ./data/ -c ./configs/mobilenetv1_bn_uniform_reset_bn.yml
