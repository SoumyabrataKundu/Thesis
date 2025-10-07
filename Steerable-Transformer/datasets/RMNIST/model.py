import os
import h5py
import torch

import Steerable.nn as snn
from Steerable.utils.hdf5 import HDF5

class Model(torch.nn.Module):
    def __init__(self, n_radius, max_m) -> None:
        super(Model, self).__init__()
        n_theta = 40
        self.num_classes = 10

        self.network = torch.nn.Sequential(
            snn.SE2ConvType1(1,  24, 5, n_radius, n_theta, max_m, padding='same'),
            snn.SE2BatchNorm(), 
            snn.SE2CGNonLinearity(max_m),
            snn.SE2BatchNorm(),
            snn.SE2ConvType2(24,  48, 5, n_radius, n_theta, max_m, padding='same'),
            snn.SE2BatchNorm(),

            snn.SE2AvgPool(2),

            snn.SE2ConvType2(48, 48, 5, n_radius, n_theta, max_m, padding='same'),
            snn.SE2BatchNorm(),
            snn.SE2CGNonLinearity(max_m),
            snn.SE2BatchNorm(),
            snn.SE2ConvType2(48, 96, 5, n_radius, n_theta, max_m, padding='same'), 
            snn.SE2BatchNorm(),

            snn.SE2AvgPool(2),

            snn.SE2ConvType2(96, 64, 7, n_radius, n_theta, max_m),

            snn.SE2NormFlatten(),
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        x = x.type(torch.cfloat)
        return self.network(x)
  

#######################################################################################################################
###################################################### Dataset ########################################################
#######################################################################################################################



Loss = torch.nn.CrossEntropyLoss()


def get_datasets(data_path):
    # Load the dataset
    data_file = h5py.File(os.path.join(data_path, 'RotMNIST.hdf5'), 'r')
    
    # Load datasets
    train_dataset = HDF5(data_file, mode='train')
    val_dataset = HDF5(data_file, mode='val')
    test_dataset = HDF5(data_file, mode='test')

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset} 
