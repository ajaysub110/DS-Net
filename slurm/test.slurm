#!/bin/bash
#SBATCH --job-name=test
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_6gb|gpu_12gb
#SBATCH --mem=64G
#SBATCH -c 3

#SBATCH --array=1-9

# activate conda env
source /misc/kcgscratch1/ajay_exp/miniconda3/etc/profile.d/conda.sh
conda activate dsnet

# run BN recalibration
python -u -m torch.distributed.launch --nproc_per_node=1 train.py ./data/ --sweep_step $SLURM_ARRAY_TASK_ID -c ./configs/mobilenetv1_bn_uniform_reset_bn.yml
