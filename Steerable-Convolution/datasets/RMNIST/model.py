import os
import h5py

import torch
import Steerable.nn as snn
from Steerable.utils import HDF5, AddGaussianNoise, RandomRotate

###################################################################################################################
################################################ Model ############################################################
###################################################################################################################

class Model(torch.nn.Module):
    def __init__(self, freq_cutoff, interpolation) -> None:
        super(Model, self).__init__()
        n_angle = 1000

        self.network = torch.nn.Sequential(
            # Convolution Block I
            snn.SE2ConvType1(1,  24, 5, freq_cutoff, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE2CGNonLinearity(24, freq_cutoff, n_angle),
            snn.SE2BatchNorm(),
            snn.SE2ConvType2(24,  48, 5, freq_cutoff, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE2BatchNorm(),

            snn.SE2AvgPool(2),

            # Convolution Block II
            snn.SE2ConvType2(48, 48, 5, freq_cutoff, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE2CGNonLinearity(48, freq_cutoff, n_angle),
            snn.SE2BatchNorm(),
            snn.SE2ConvType2(48, 96, 5, freq_cutoff, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE2BatchNorm(),

            snn.SE2AvgPool(2),

            # Flattening and Invariant Layers
            snn.SE2ConvType2(96, 64, 7, freq_cutoff, n_angle=n_angle, interpolation_type=interpolation),
            snn.SE2NormFlatten(),

            # MLP Block
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x.type(torch.cfloat))


###################################################################################################################
############################################## Dataset ############################################################
################################################################################################################### 

def get_datasets(data_path, rotate=True, noise=0):
    if rotate:
        data_file = h5py.File(os.path.join(data_path, 'RotMNIST.hdf5'), 'r')
        train_dataset = RandomRotate(HDF5(data_file, mode='train'))
    else:
        data_file = h5py.File(os.path.join(data_path, 'MNIST.hdf5'), 'r')
        train_dataset = HDF5(data_file, mode='train')

    data_file = h5py.File(os.path.join(data_path, 'RotMNIST.hdf5'), 'r')
    test_dataset = HDF5(data_file, mode='test')

    if noise>0:
        test_dataset = AddGaussianNoise(test_dataset, noise)

    return {'train' : train_dataset, 'val' : None, 'test' : test_dataset} 
