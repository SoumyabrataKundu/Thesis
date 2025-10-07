#!/bin/bash

#SBATCH --job-name=RUNDATASETfFREQ
#SBATCH --output=output.sen
#SBATCH --error=error.sen
#SBATCH --account=pi-risi
#SBATCH --partition=gm4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load python
source activate /home/soumyabratakundu/.conda/envs/conda_env

start_time=`date +%s`

python sensitivity.py --model_path="./" --data_path="DATAPATH" --batch_size=BATCHSIZE --freq_cutoff=FREQ --interpolation=ORDER

end_time=`date +%s`
runtime=$((end_time - start_time))
echo Runtime = $((runtime / 3600))h $(((runtime % 3600) / 60))m
