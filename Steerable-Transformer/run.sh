#!/bin/bash

# Experiment
models=(1)
runs=(1 2 3 4) 
datasets=("MoNuSeg")
rotate=(0 1)
script="train"
metric_type="dice"
save=0

# Model Hyperparameters
batch_size=3
epochs=100

# Job Parameters
main_directory=$PWD
data_path="/project2/risi/soumyabratakundu/Data"
jobs_done=0
MAX_CONCURRENT_JOBS=28

wait_for_jobs() {
    while [ $(squeue -u $USER | tail -n +2 | wc -l) -ge $MAX_CONCURRENT_JOBS ]; do
        sleep 60
    done
}

on_interrupt() {
    echo Script interrupted. Jobs submitted so far: $((job_counter-1)) / $total_jobs"."
    exit
}
stty -echoctl
trap on_interrupt SIGINT

mkdir final_runs1 2>/dev/null
cd final_runs1

job_counter=0

for data in "${datasets[@]}"
do
    mkdir ${data} 2>/dev/null
    cd ${data}

    for model in "${models[@]}"
    do

        for rot in "${rotate[@]}"
        do

            for sim in "${runs[@]}"
            do
                ((job_counter++))
                if [ "$job_counter" -le "$jobs_done" ]; then
                     echo job ${job_counter} run${run} ${data} already submitted.
                     continue
                fi

                run=$((10*model + 5*rot + sim))
                if [ $run -lt 10 ]; then
                     run=0$run
                fi
                mkdir run$run 2>/dev/null
                cd run$run

                ## Copy Files
                if [ ${script} == "train" ]; then
                    cp -r ${main_directory}/../Steerable/Steerable/ ./
                    cp ${main_directory}/datasets/${data}/model${model}.py ./model.py
                fi
                cp ${main_directory}/scripts/${script}.sh ./ 2>/dev/null
                cp ${main_directory}/scripts/${script}.py ./ 2>/dev/null
      
                ## Modify Script
                sed -i "s/RUN/${run}/g" ${script}.sh
                sed -i "s/ROTATE/${rot}/g" ${script}.sh
                sed -i "s/DATASET/${data:0:3}/g" ${script}.sh
                sed -i "s/LOSS/${loss}/g" ${script}.sh
                sed -i "s#DATAPATH#${data_path}/${data}/data#g" ${script}.sh
                sed -i "s/BATCHSIZE/${batch_size}/g" ${script}.sh
                sed -i "s/EPOCHS/${epochs}/g" ${script}.sh
                sed -i "s/METRICTYPE/${metric_type}/g" ${script}.sh
                sed -i "s/SAVE/${save}/g" ${script}.sh

                wait_for_jobs

                echo job ${job_counter} ${script}-run${run} ${data}
                sbatch ${script}.sh

                sed -i "s/^jobs_done=.*/jobs_done=$((job_counter))/g" $main_directory/run.sh

                cd ../
           done
        done
    done
    cd ../ 
done

sed -i "s/^jobs_done=.*/jobs_done=0/g" $main_directory/run.sh
echo "All jobs Submitted!"
echo
