#!/bin/bash

#SBATCH --job-name=Jupyter
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --account=pi-risi
#SBATCH --partition=gm4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0

module load python
source activate /home/soumyabratakundu/.conda/envs/conda_env

HOST_IP=`/sbin/ip route get 8.8.8.8 | awk '{print $7;exit}'`
jupyter-notebook --no-browser --ip=$HOST_IP
