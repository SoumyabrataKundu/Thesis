#!/bin/bash

#SBATCH --job-name=EvRUN-DATASET
#SBATCH --output=output.eval
#SBATCH --error=error.eval
#SBATCH --account=pi-risi
#SBATCH --partition=gm4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load python
source activate /home/soumyabratakundu/.conda/envs/conda_env

start_time=`date +%s`

python eval.py --model_path="./" --data_path="DATAPATH" --batch_size=BATCHSIZE --save=SAVE

end_time=`date +%s`
runtime=$((end_time - start_time))
echo
echo Runtime = $((runtime / 3600))h $(((runtime % 3600) / 60))m
