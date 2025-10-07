import os
import re
import numpy as np
import pickle
from termcolor import colored

import warnings
warnings.filterwarnings('ignore')

def main(root, dataset=None, save=False):
    datasets = os.listdir(root) if not dataset else [dataset]
    
    for dataset in datasets:
        path = os.path.join(root, dataset)
        pattern = re.compile(r"^run(-?\d+)$")

        n_sim   = np.zeros((4, 2), dtype=np.int64)    # model, rotated
        running = np.zeros((4, 2), dtype=np.int64)    # model, rotated
        dice    = np.zeros((4, 2, 2, 2))              # model, rotated, micro/macro, mean/var
        pattern1 = re.compile(r"Micro Dice\s*=\s*(\d+\.?\d*)\s*")
        pattern2 = re.compile(r"Macro Dice\s*=\s*(\d+\.?\d*)\s*")

        for d in os.listdir(path):
            match = pattern.match(d)
            if match:
                run = int(match.group(1))
                model = (run-1) // 10
                rotated = ((run-1) % 10) // 5
            try:
                f = 'output.eval' if dataset not in ['RMNIST', 'ModelNet10'] else 'output.train'
                with open(os.path.join(path, d, f), "r") as f:
                   for line in f:
                        match1 = pattern1.match(line)
                        match2 = pattern2.match(line)
                        if match1:
                            dice[model, rotated, 0, 0] += float(match1.group(1))
                            dice[model, rotated, 0, 1] += float(match1.group(1)) ** 2
                            n_sim[model, rotated] += 1
                        if match2:
                            dice[model, rotated, 1, 0] += float(match2.group(1))
                            dice[model, rotated, 1, 1] += float(match2.group(1)) ** 2

            except Exception as e:
                if os.path.exists(os.path.join(path, d, 'output.train')):
                    running[model, rotated] += 1

        n_sim = np.expand_dims(n_sim, axis=-1)
        final_stats = np.zeros((4, 2, 2, 2))       # model, rotated, macro/micro, mean/var
        final_stats[...,0] = dice[...,0] / n_sim
        final_stats[...,1] = 1.28 * np.sqrt((dice[...,1]-(dice[...,0]**2/n_sim)) / ((n_sim-1)*n_sim))
        final_stats = final_stats * 100
        n_sim = n_sim[...,0]
        print(colored(f'\nResults for {dataset} Dataset', 'cyan'))
        for model in range(len(final_stats)):
            print(colored(f'\t\tModel {model} : ', 'green'), end='')
            for rot in [0,1]:
                print(colored(f"{'Non-' if rot==0 else ''}Rotated : ", 'red') + \
                      colored(f'{final_stats[model, rot, 0, 0]:.2f} +- {final_stats[model, rot, 0, 1]:.2f} ', 'white') + \
                      colored(' / ', 'yellow') + \
                      colored(f'{final_stats[model, rot, 1, 0]:.2f} +- {final_stats[model, rot, 1, 1]:.2f} ', 'white') + \
                      colored(f'({n_sim[model, rot]}{" + " + str(running[model, rot]) if running[model, rot]>0 else ""})\t', 'dark_grey'), end='')
            print()
        if save:
            with open(os.path.join('stats/', dataset + "_stats.pkl"), 'wb') as f:
                pickle.dump(final_stats, f) 
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, default="archive/final_runs/")
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("-s", "--save", action="store_true")

    args = parser.parse_args()
    main(**args.__dict__)
