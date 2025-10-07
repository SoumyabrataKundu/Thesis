#!/bin/bash


#SBATCH --job-name=TeRUN-DATASET
#SBATCH --output=output.test
#SBATCH --error=error.test
#SBATCH --account=pi-risi
#SBATCH --partition=gm4
#SBATCH --nodes=1
#SBATCH --gres=gpu:GPU


module load python
source activate /home/soumyabratakundu/.conda/envs/conda_env

start_time=`date +%s`

python train.py --data_path="DATAPATH" --batch_size=BATCHSIZE --n_radius=RADIUS --max_m=THETA --num_epochs=0 --metric_type=METRICTYPE --save=SAVE

end_time=`date +%s`
runtime=$((end_time - start_time))
echo
echo Runtime = $((runtime / 3600))h $(((runtime % 3600) / 60))m
