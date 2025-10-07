import os
import re
import numpy as np
import pickle
from termcolor import colored

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

def main(root, dataset=None, save=False):
    datasets = [dataset] if dataset else os.listdir(root)
    for dataset in datasets:
        path = os.path.join(root, dataset)
        directories = []
        order_values = set()
        k_values = set()

        pattern = re.compile(r"^run(-?\d+)$")
        for d in os.listdir(path):
            if os.path.isdir(os.path.join(path, d)):
                match = pattern.match(d)
                if match:
                    run = int(match.group(1))
                    order_values.add((run-1) // 10)
                    directories.append((d, run))

        pattern = re.compile(r"^f(\d+)$")
        for directory, _ in directories:
            for d in os.listdir(os.path.join(path, directory)):
                match = pattern.match(d)
                if match:
                    k_values.add(int(match.group(1)))

        order_values = sorted(list(order_values))
        k_values = sorted(list(k_values))

        pattern = re.compile(r"Testing Accuracy\s*=\s*(\d+\.?\d*)\s*%")

        n_sim       = np.zeros((len(order_values), len(k_values), 2, 2), dtype=np.int64)
        running     = np.zeros((len(order_values), len(k_values), 2, 2), dtype=np.int64)
        accuracy    = np.zeros((len(order_values), len(k_values), 2, 2))
        sensitivity = np.zeros((len(order_values), len(k_values), 8, 2, 3))
        for directory, run in directories:
            for k in range(len(k_values)):
                d = os.path.join(path, directory, f'f{k_values[k]}')
                if os.path.isdir(d):
                    order = order_values.index((run-1) // 10)
                    rotated = int((run-1)%10 >= 5)
                    # Accuracy
                    try:
                        with open(os.path.join(d, 'output'), "r") as f:
                            running[order, k, 0, rotated] += 1
                            for line in f:
                                match = pattern.match(line)
                                if match:
                                    accuracy[order, k, rotated, 0] += float(match.group(1))
                                    accuracy[order, k, rotated, 1] += float(match.group(1)) ** 2
                                    n_sim[order, k, 0, rotated] += 1
                                    running[order, k, 0, rotated] -= 1
                                    break
                    except FileNotFoundError:
                        pass
                        #print(f"  {os.path.join(d, 'output')} not found.")
                    # Sensitivity
                    try:
                        running[order, k, 1, rotated] += os.path.isfile(os.path.join(d, 'output.sen'))
                        with open(os.path.join(d, 'log/sensitivity.pkl'), 'rb') as f:
                            loaded = pickle.load(f)
                            try:
                                sensitivity[order, k, :, rotated, 0] += loaded[...,0]
                                sensitivity[order, k, :, rotated, 1] += loaded[...,0] ** 2
                                sensitivity[order, k, :, rotated, 2] += loaded[...,1]
                            except:
                                raise ValueError(f'{d}, {run}')                    
                            n_sim[order, k, 1, rotated] += 1
                            running[order, k, 1, rotated] -= 1

                    except FileNotFoundError:
                        pass
                        #print(f"    {os.path.join(d, 'log/sensitivity.pkl')} not found.")
                else:
                    pass
                    #print(f"{d} not found.")
   
        final_stat = np.zeros((len(order_values), len(k_values), 9, 2, 2))

        # Accuracy
        final_stat[...,0,:,0] = accuracy[..., 0] / n_sim[...,0,:]
        final_stat[...,0,:,1] = 1.28 * np.sqrt((accuracy[..., 1] -  (accuracy[..., 0] **2 / n_sim[...,0,:])) / ((n_sim[...,0,:]-1)*n_sim[...,0,:]))

        # Sensitivity
        n_sim_sen = np.expand_dims(n_sim[...,1,:], axis=-2)
        final_stat[...,1:,:,0] = sensitivity[..., 0] / n_sim_sen
        final_stat[...,1:,:,1] = 1.28 * np.sqrt((sensitivity[..., 1] - (sensitivity[..., 0]**2 / n_sim_sen)) / ((n_sim_sen-1)*n_sim_sen)  + sensitivity[..., 2] / (n_sim_sen**2))

        print()
        print(colored(f"Results for {dataset} : ", 'cyan'))
        for index, order in enumerate(order_values):
            print(colored(f'\tInterpolation Order {order}:', 'yellow'))
            for k in range(len(k_values)):
                print(colored(f'\t\tAccuracy of f{k_values[k]: <2} : ', 'green'), end="")
                for r in [0,1]:
                    print(colored(f'{"Not " if r==0 else ""}Rotated = ', 'red') + \
                          colored(f'{final_stat[index, k, 0, r, 0]:.2f} +- {final_stat[index, k, 0, r, 1]:.2f} % ', 'white') + \
                          colored(f'({n_sim[index, k, 0, r]}{" + " + str(running[index, k, 0, r]) if running[index, k, 0, r]>0 else ""})', 'dark_grey') + \
                          colored(f'({n_sim[index, k, 1, r]}{" + " + str(running[index, k, 1, r]) if running[index, k, 1, r]>0 else ""})', 'dark_grey'), end=" ")
                print()
        if save:
            with open(os.path.join('stats/', dataset + "_stats.pkl"), 'wb') as f:
                pickle.dump(final_stat, f)
    print()
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, default="experiment_runs/")
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("-s", "--save", action="store_true")

    args = parser.parse_args()
    main(**args.__dict__)
