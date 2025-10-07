import torch
import numpy as np
import h5py
import os
import fnmatch
import sys

sys.path.append('../../Steerable/')
from Steerable.datasets.hdf5 import HDF5Dataset

# Dataset Generation
class RotMNIST(torch.utils.data.Dataset):
    def __init__(self, data_path, mode) -> None:
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid mode {mode}. Should be one of train or test.')
        
        self.data = np.load(os.path.join(data_path, 'rotated_'+ mode + '.npz'))
    def __getitem__(self, index):
        image = self.data['x'][index].reshape(1, 28, 28)    
        label = self.data['y'][index]
        return image, label

    def __len__(self):
        return len(self.data['x'])

def main(data_path):
    filename = 'RotMNIST.hdf5'
    hdf5file = HDF5Dataset(filename)

    for mode in ['train', 'val', 'test']:
        dataset = RotMNIST(data_path=data_path, mode=mode)
        hdf5file.create_hdf5_dataset(mode, dataset)
    
    
if __name__== '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/')

    args = parser.parse_args()
    main(**args.__dict__)
