import os
import h5py

import torch
import Steerable.nn as snn
from Steerable.utils import HDF5, RandomRotate

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        
        device='cuda' 
        n_angle = 256
        self.num_classes = 4
        ratio = [8, 4, 1]
        channel = lambda factor: [x * factor for x in ratio]

        encoder_dim = channel(16)
        decoder_dim = channel(16)

        self.convolution_stem1 = torch.nn.Sequential(
            snn.SE3Conv(4, channel(1), 7, n_angle=n_angle, stride=2),
            snn.SE3NormNonLinearity(channel(1)),
            snn.SE3Conv(channel(1), channel(2), 5, n_angle=n_angle, padding='same'),
            snn.SE3BatchNorm(),
        )
        
        self.pool1 = snn.SE3AvgPool(4)
  
        self.convolution_stem2 = torch.nn.Sequential(
            snn.SE3Conv(channel(2), channel(4), 5, n_angle=n_angle, padding='same'),
            snn.SE3NormNonLinearity(channel(4)),
            snn.SE3Conv(channel(4), encoder_dim, 5, n_angle=n_angle, padding='same'),
            snn.SE3BatchNorm(),
        )

        self.pool2 = snn.SE3AvgPool(4)

        self.encoder_decoder = torch.nn.Sequential(
            snn.SE3PositionwiseFeedforward(channel(16), channel(32)),
        )
        
        self.convolution_head1 = torch.nn.Sequential(
            snn.SE3Conv(channel(32), channel(4), 5, n_angle=n_angle, padding = 'same'),
            snn.SE3NormNonLinearity(channel(4)),
            snn.SE3Conv(channel(4), channel(2), 5, n_angle=n_angle, padding = 'same'),
            snn.SE3BatchNorm(),
        )

        self.convolution_head2 = torch.nn.Sequential(
            snn.SE3Conv(channel(4), channel(1),5, n_angle=n_angle, padding = 'same'),
            snn.SE3NormNonLinearity(channel(1)),
            snn.SE3BatchNorm(),
	    snn.SE3Conv(channel(1),self.num_classes,5, n_angle=n_angle, padding = 'same'),
        )
       
        
    def forward(self, x):
        x = x.type(torch.cfloat)
        x_shape = x.shape

        # Downsampling
        ## Downsample I
        stem1 = self.convolution_stem1(x)
        x = self.pool1(stem1)
        ## Downsample II
        stem2 = self.convolution_stem2(x)
        x = self.pool2(stem2)

        # Encoder-Decoder
        x = self.encoder_decoder(x)

        # Upsampling
        ## Upsample I
        x, channels = snn.merge_channel_dim(x)
        x = torch.nn.functional.interpolate(x.real, size=stem2[0].shape[-3:], mode="trilinear") + \
                  1j *torch.nn.functional.interpolate(x.imag, size=stem2[0].shape[-3:], mode="trilinear")
        x = snn.split_channel_dim(x, channels=channels)
        x = [torch.cat([x[l], stem2[l]], dim=2) for l in range(len(x))] # skip connection
        x = self.convolution_head1(x)
        ## Upsample II
        x, channels = snn.merge_channel_dim(x)
        x = torch.nn.functional.interpolate(x.real, size=stem1[0].shape[-3:], mode="trilinear") + \
                  1j * torch.nn.functional.interpolate(x.imag, size=stem1[0].shape[-3:], mode="trilinear")
        x = snn.split_channel_dim(x, channels=channels)
        x = [torch.cat([x[l], stem1[l]], dim=2) for l in range(len(x))] # skip connection
        x = self.convolution_head2(x)
        ## Upsampling III
        x = x[0].squeeze(1)
        x = torch.nn.functional.interpolate(x.real, size=x_shape[-3:], mode="trilinear") + \
                  1j * torch.nn.functional.interpolate(x.imag, size=x_shape[-3:], mode="trilinear")

        # Logit Scores
        x = x.abs()

        return x

#######################################################################################################################
###################################################### Dataset ########################################################
####################################################################################################################### 

def get_datasets(data_path, rotate = True):
    data_file = h5py.File(os.path.join(data_path, 'Brain_patched.hdf5'), 'r')
    train_dataset = HDF5(data_file, mode='train')
    if rotate:
        train_dataset = RandomRotate(train_dataset)
    val_dataset = HDF5(data_file, mode='val')
    test_dataset = HDF5(data_file, mode='test')

    data_file = h5py.File(os.path.join(data_path, 'Brain.hdf5'))
    eval_val_dataset = HDF5(data_file, mode='val')
    eval_test_dataset = HDF5(data_file, mode='test')

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset, 'eval_val' : eval_val_dataset, 'eval_test' : eval_test_dataset}
