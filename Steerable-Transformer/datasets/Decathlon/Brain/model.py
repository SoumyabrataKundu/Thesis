import torch
import os
import h5py

import torch

from Steerable.nn import *
from Steerable.datasets.hdf5 import HDF5
from Steerable.Segmentation.loss import SegmentationLoss

class Model(torch.nn.Module):
    def __init__(self, n_radius, max_m) -> None:
        super(Model, self).__init__()
        
        device='cuda' 
        n_theta = 40
        self.num_classes = 4
        encoder_dim = [64,32,16]
        decoder_dim = [64,32,16]

        self.convolution_stem1 = nn.Sequential(
            SE3Conv(4,[8,4,2],7, n_radius, n_theta, stride=2),
            SE3NormNonLinearity([8,4,2]),
            SE3Conv([8,4,2],[16,8,4],7, n_radius, n_theta, padding='same'),
            SE3BatchNorm(),
        )
        
        self.pool1 = SE3AvgPool(8)
  
        self.convolution_stem2 =  nn.Sequential(
            SE3Conv([16,8,4],[32,16,8],5, n_radius, n_theta, padding='same'),
            SE3CGNonLinearity([32,16,8]),
            SE3Conv([32,16,8],encoder_dim,5, n_radius, n_theta, padding='same'),
            SE3BatchNorm(),
        )

        self.pool2 = SE3AvgPool(4)

        self.encoder = SE3TransformerEncoder(encoder_dim, 4, n_layers = 2, add_pos_enc=True)
        
        self.convolution_head1 = nn.Sequential(
            SE3Conv(decoder_dim,[32,16,8],5, n_radius, n_theta, padding = 'same'),
            SE3NormNonLinearity([32,16,8]),
            SE3Conv([32,16,8],[16,8,4],5, n_radius, n_theta, padding = 'same'),
            SE3BatchNorm(),
        )

        self.convolution_head2 = nn.Sequential(
            SE3Conv([16,8,4],[8,4,2],7, n_radius, n_theta, padding = 'same'),
            SE3NormNonLinearity([8,4,2]),
            SE3BatchNorm(),
	    SE3Conv([8,4,2],self.num_classes,7, n_radius, n_theta, padding = 'same'),
        )
       
        
    def forward(self, x):
        x_shape = x.shape
        x = x.type(torch.cfloat)
        
        # Downsampling
        stem1 = self.convolution_stem1(x)
        x = self.pool1(stem1)
        stem2 = self.convolution_stem2(x)
        x = self.pool2(stem2)

        # Encoder
        x = self.encoder(x)

        # Upsampling
        x, channels = merge_channel_dim(x)
        x = nn.functional.interpolate(x.real, size=stem2[0].shape[-3:], mode="trilinear") + \
                  1j * nn.functional.interpolate(x.imag, size=stem2[0].shape[-3:], mode="trilinear")
        x = split_channel_dim(x, channels=channels)
        x = [x[l] + stem2[l] for l in range(len(x))] # skip connection
        x = self.convolution_head1(x)

        x, channels = merge_channel_dim(x)
        x = nn.functional.interpolate(x.real, size=stem1[0].shape[-3:], mode="trilinear") + \
                  1j * nn.functional.interpolate(x.imag, size=stem1[0].shape[-3:], mode="trilinear")
        x = split_channel_dim(x, channels=channels)
        x = [x[l] + stem1[l] for l in range(len(x))] # skip connection
        x = self.convolution_head2(x)

        x = x[0].squeeze(1)
        x = nn.functional.interpolate(x.real, size=x_shape[-3:], mode="trilinear") + \
                  1j * nn.functional.interpolate(x.imag, size=x_shape[-3:], mode="trilinear")
        return x.abs()

#######################################################################################################################
###################################################### Dataset ########################################################
####################################################################################################################### 

Loss=SegmentationLoss(loss_type='Focal')

def get_datasets(data_path):
    # Load the dataset
    data_file = h5py.File(os.path.join(data_path, 'Brain.hdf5'), 'r')

    # Load datasets
    datasets = HDF5(data_file)
    train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(datasets, [0.05, 0.05, 0.05, 0.85])

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}
