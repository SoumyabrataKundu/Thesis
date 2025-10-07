import os
import h5py

import torch
import Steerable.nn as snn
from Steerable.utils import HDF5, AddGaussianNoise, RandomRotate

###################################################################################################################
############################################## Model ##############################################################
###################################################################################################################

device = torch.device('cuda')
class Model(torch.nn.Module):
    def __init__(self, freq_cutoff, interpolation) -> None:
        super().__init__()
        n_angle = 256
        ratio = [2**(freq_cutoff-l) for l in range(freq_cutoff+1)]
        channel = lambda factor: [x * factor for x in ratio]

        self.network = torch.nn.Sequential(
            snn.SE3Conv(1, channel(1), 5, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE3CGNonLinearity(channel(1)),
            snn.SE3BatchNorm(),
            snn.SE3Conv(channel(1), channel(1), 5, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE3BatchNorm(),

            snn.SE3AvgPool(2),

            snn.SE3Conv(channel(1), channel(2), 5, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE3CGNonLinearity(channel(2)),
            snn.SE3BatchNorm(),
            snn.SE3Conv(channel(2), channel(2), 5, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE3BatchNorm(),

            snn.SE3AvgPool(2),

            snn.SE3Conv(channel(2), channel(4), 5, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE3CGNonLinearity(channel(4)),
            snn.SE3BatchNorm(),
            snn.SE3Conv(channel(4), channel(4), 5, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE3BatchNorm(),

            snn.SE3AvgPool(2),

            snn.SE3Conv(channel(4), channel(8), 5, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE3CGNonLinearity(channel(8)),
            snn.SE3BatchNorm(),
            snn.SE3Conv(channel(8), channel(8), 5, n_angle=n_angle, padding='same', interpolation_type=interpolation),
            snn.SE3BatchNorm(),

            snn.SE3Conv(channel(8), 512, 4, n_angle=n_angle, interpolation_type=interpolation),

            snn.SE3NormFlatten(),

            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(128, 55),
        )
         
    def forward(self,x):
        return self.network(x.type(torch.cfloat))

###################################################################################################################
############################################## Dataset ############################################################
###################################################################################################################

def get_datasets(data_path, rotate=True, noise=0):
    if rotate:
        data_file = h5py.File(os.path.join(data_path, 'SHREC17_perturbed32.hdf5'), 'r')
        train_dataset = RandomRotate(HDF5(data_file, mode='train'))
        val_dataset = HDF5(data_file, mode='val')
    else:
        data_file = h5py.File(os.path.join(data_path, 'SHREC17_normal32.hdf5'), 'r')
        train_dataset = HDF5(data_file, mode='train')
        val_dataset = HDF5(data_file, mode='val')

    data_file = h5py.File(os.path.join(data_path, 'SHREC17_perturbed32.hdf5'), 'r')
    test_dataset = HDF5(data_file, mode='test')

    if noise > 0:
        test_dataset = AddGaussianNoise(test_dataset, noise)

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}
