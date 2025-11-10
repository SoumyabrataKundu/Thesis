import os
import h5py
import time
from numpy import mean, std

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import Steerable.nn as snn


###################################################################################################################
############################################## Model ##############################################################
###################################################################################################################

device = torch.device('cuda')
class Model(torch.nn.Module):
    def __init__(self, n_radius, maxl, interpolation, restricted) -> None:
        super().__init__()
        n_theta = 40
        restricted = bool(restricted)
        conv_first = False
        ratio = [2**(maxl-l) for l in range(maxl+1)]
        channel = lambda factor: [x * factor for x in ratio]

        self.network = nn.Sequential(
            snn.SE3Conv(1, channel(1), 5, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),
            snn.SE3CGNonLinearity(channel(1)),
            snn.SE3BatchNorm(),
            snn.SE3Conv(channel(1), channel(1), 3, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),
            snn.SE3BatchNorm(),

            snn.SE3AvgPool(2),

            snn.SE3Conv(channel(1), channel(2), 3, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),
            snn.SE3CGNonLinearity(channel(2)),
            snn.SE3BatchNorm(),
            snn.SE3Conv(channel(2), channel(2), 3, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),
            snn.SE3BatchNorm(),

            snn.SE3AvgPool(2),

            snn.SE3Conv(channel(2), channel(4), 3, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),
            snn.SE3CGNonLinearity(channel(4)),
            snn.SE3BatchNorm(),
            snn.SE3Conv(channel(4), channel(4), 3, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),
            snn.SE3BatchNorm(),

            snn.SE3AvgPool(2),

            snn.SE3Conv(channel(4), channel(8), 3, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),
            snn.SE3CGNonLinearity(channel(8)),
            snn.SE3BatchNorm(),
            snn.SE3Conv(channel(8), channel(8), 3, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),
            snn.SE3BatchNorm(),
            #snn.SE3TransformerEncoder(channel(8), 8, n_layers=2, add_pos_enc=True),

            snn.SE3Conv(channel(8), 512, 4, n_radius=n_radius, n_theta=n_theta, restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),

            snn.SE3NormFlatten(),

            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(128, 10),
        )
         
    def forward(self,x):
        return self.network(x.type(torch.cfloat))

###################################################################################################################
############################################## Dataset ############################################################
###################################################################################################################


class AddGaussianNoise:
    def __init__(self, sd: float):
        self.sd = sd

    def __call__(self, tensor: torch.Tensor):
        return tensor + (torch.randn_like(tensor) * self.sd)

def get_datasets(data_path, noise=0):

    # Transformations
    if noise>0:
        image_transform = transforms.Compose([
            transforms.Normalize(mean=0, std = 1),
            AddGaussianNoise(noise),
            transforms.Normalize(mean=0, std = 1),
            ])
    else:
        image_transform = transforms.Compose([
            transforms.Normalize(mean=0, std = 1)
            ])
    # Load the dataset
    data_file = h5py.File(os.path.join(data_path, 'ModelNet10_rotate32.hdf5'), 'r')

    # Load datasets
    train_dataset = RotMNIST(data_file, mode='train', image_transform=image_transform)
    val_dataset = None
    test_dataset = RotMNIST(data_file, mode='test', image_transform=image_transform)

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}
