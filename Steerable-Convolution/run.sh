#!/bin/bash

# Hyperparameters
runs=(1 2 3 4 5)
datasets=("ModelNet10" "SHREC17")
freq_cutoff=(0 1 2 3)
rotate=(0 1)
interpolations=(0 1 -1)
restore=0
total_jobs=$((${#runs[@]} * ${#interpolations[@]} * ${#rotate[@]} * ${#freq_cutoff[@]} * ${#datasets[@]}))

# Job Parameters
script="sensitivity"
main_directory=$PWD
jobs_done=0
MAX_CONCURRENT_JOBS=28
data_path="../Data"

wait_for_jobs() {
    while [ $(squeue -u $USER | tail -n +2 | wc -l) -ge $MAX_CONCURRENT_JOBS ]; do
        sleep 60
    done
}

# Training Parameters
epochs=150
get_batch_size() {
    local k=$1
    if [ "$k" -eq 3 ]; then
        echo 85
    else
        echo 100
    fi
}

on_interrupt() {
    echo Script interrupted. Jobs submitted so far: $((job_counter-1)) / $total_jobs"."
    exit
}
stty -echoctl
trap on_interrupt SIGINT

mkdir experiment_runs 2>/dev/null
cd experiment_runs

echo 
job_counter=0

for dataset in "${datasets[@]}"
do
    mkdir ${dataset} 2>/dev/null
    cd ${dataset} 

    for order in "${interpolations[@]}"
    do
        for rot in "${rotate[@]}"
        do
            for sim in "${runs[@]}"
            do

                run=$((10*order + 5*rot + sim))
                if [ $run -ge 0 ] && [ $run -le 9 ]; then
                    run="0$run"
                fi

                mkdir run${run} 2>/dev/null
                cd run${run}

                for freq in "${freq_cutoff[@]}"
                do
                    ((job_counter++))
                    if [ "$job_counter" -le "$jobs_done" ]; then
                        if [ "$job_counter" -eq "$jobs_done" ]; then
                            echo Last job submitted - job $(($job_counter / 10))$(($job_counter%10)) / $total_jobs : ${dataset} ${script}-run "#"${run} freq-cutoff=$(($freq / 10))$(($freq%10)) order=${order}
                            echo
                        fi
                        continue
                    fi

                    mkdir f${freq} 2>/dev/null
                    cd f${freq}

                    if [ ${script} == "train" ] && [ ${restore} -eq 0 ] && [ ${epochs} -gt 0 ]; then
                            mkdir Steerable 2>/dev/null
                            cp -r ${main_directory}/../Steerable/Steerable/ .

                            cp ${main_directory}/datasets/${dataset}/model.py ./
                    fi

                    if [ ${script} == "eval" ]; then
                        cp -r ${main_directory}/datasets/${dataset}/evaluator/ ./ 2>/dev/null
                        cp -r ${main_directory}/datasets/${dataset}/eval.sh ./ 
                        cp -r ${main_directory}/datasets/${dataset}/eval.py ./ 
                    fi

                    cp ${main_directory}/scripts/${script}.sh ./ 2>/dev/null
                    cp ${main_directory}/scripts/${script}.py ./ 2>/dev/null
 
                    sed -i "s/RUN/${run}/g" ${script}.sh
                    sed -i "s/DATASET/${dataset:0:3}/g" ${script}.sh
                    sed -i "s/FREQ/${freq}/g" ${script}.sh
                    sed -i "s/ORDER/${order}/g" ${script}.sh
                    sed -i "s/ROTATE/${rot}/g" ${script}.sh
                    sed -i "s#DATAPATH#${data_path}/${dataset}/data#g" ${script}.sh
                    sed -i "s/BATCHSIZE/$(get_batch_size "$freq")/g" ${script}.sh
                    sed -i "s/EPOCHS/${epochs}/g" ${script}.sh
                    sed -i "s/RESTORE/${restore}/g" ${script}.sh
 
                    wait_for_jobs

                    echo job $(($job_counter / 10))$(($job_counter%10)) / $total_jobs : ${dataset} ${script}-run "#"${run} freq-cutoff=$(($freq / 10))$(($freq%10)) order=${order} at $(date +"%H:%M:%S") 
                    sbatch ${script}.sh
                    sed -i "s/^jobs_done=.*/jobs_done=$((job_counter))/g" $main_directory/run.sh

                    cd ../
                done
                cd ../
            done
        done
    done
    cd ../
done


sed -i "s/^jobs_done=.*/jobs_done=0/g" $main_directory/run.sh
echo "All jobs Submitted!"
echo
