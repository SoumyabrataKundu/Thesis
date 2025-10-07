import os
import h5py
import torch

import Steerable.nn as snn
from Steerable.utils import HDF5, RandomRotate

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        n_angle = 1000
        freq_cutoff = 8
        self.num_classes = 2
        transformer_dim = 139

        self.convolution_stem1 = torch.nn.Sequential(
            snn.SE2ConvType1(3,8,5, freq_cutoff, n_angle=n_angle, padding='same'),
            snn.SE2NormNonLinearity(8, freq_cutoff),
            snn.SE2ConvType2(8,16,5, freq_cutoff, n_angle=n_angle, padding='same'),
            snn.SE2BatchNorm(),
        )
        
        self.pool1 = snn.SE2AvgPool(4)
  
        self.convolution_stem2 = torch.nn.Sequential(
            snn.SE2ConvType2(16,32,5, freq_cutoff, n_angle=n_angle, padding='same'),
            snn.SE2NormNonLinearity(32, freq_cutoff),
            snn.SE2ConvType2(32,transformer_dim,5, freq_cutoff, n_angle=n_angle, padding='same'),
            snn.SE2BatchNorm(),
        )

        self.pool2 = snn.SE2AvgPool(4)

        self.encoder_decoder = torch.nn.Sequential(
            snn.SE2PositionwiseFeedforward(transformer_dim, 2*transformer_dim, freq_cutoff),
            snn.SE2BatchNorm(),
        )
 
        self.convolution_head1 = torch.nn.Sequential(
            snn.SE2ConvType2(2*transformer_dim,32,5, freq_cutoff, n_angle=n_angle, padding = 'same'),
            snn.SE2NormNonLinearity(32, freq_cutoff),
            snn.SE2ConvType2(32,16,5, freq_cutoff, n_angle=n_angle, padding = 'same'),
            snn.SE2BatchNorm(),
        )

        self.convolution_head2 = torch.nn.Sequential(
            snn.SE2ConvType2(2*16,8,5, freq_cutoff, n_angle=n_angle, padding = 'same'),
            snn.SE2NormNonLinearity(8, freq_cutoff),
            snn.SE2BatchNorm(),
            snn.SE2ConvType2(8, self.num_classes,5, freq_cutoff, n_angle=n_angle, padding = 'same'),
        )
        
    def forward(self, x):
        x = x.type(torch.cfloat)
        
        # Downsampling
        stem1 = self.convolution_stem1(x)
        x = self.pool1(stem1)
        stem2 = self.convolution_stem2(x)
        x = self.pool2(stem2)

        # Encoder-Decoder
        x = self.encoder_decoder(x)

        # Upsampling
        x = torch.nn.functional.interpolate(x.real, size=(x.shape[-3], *stem2.shape[-2:]), mode="trilinear") + \
                  1j * torch.nn.functional.interpolate(x.imag, size=(x.shape[-3], *stem2.shape[-2:]), mode="trilinear")
        x = self.convolution_head1(torch.cat([x, stem2], dim=2)) # skip connection

        x = torch.nn.functional.interpolate(x.real, size=(x.shape[-3], *stem1.shape[-2:]), mode="trilinear") + \
                  1j * torch.nn.functional.interpolate(x.imag, size=(x.shape[-3], *stem1.shape[-2:]), mode="trilinear")
        x = self.convolution_head2(torch.cat([x, stem1], dim=2)) # skip connection
 
        # Norm
        x = torch.linalg.vector_norm(x, dim=1) 

        return x

#######################################################################################################################
###################################################### Dataset ########################################################
#######################################################################################################################

def get_datasets(data_path, rotate=True):
    data_file = h5py.File(os.path.join(data_path, 'MoNuSeg_patched256_128.hdf5'), 'r')

    train_dataset = HDF5(data_file, mode='train')
    if rotate:
        train_dataset = RandomRotate(train_dataset)
    val_dataset = HDF5(data_file, mode='val')
    test_dataset = HDF5(data_file, mode='test')
    data_file = h5py.File(os.path.join(data_path, 'MoNuSeg.hdf5'))
    eval_val_dataset = HDF5(data_file, mode='val')
    eval_test_dataset = HDF5(data_file, mode='test')

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset, 'eval_val' : eval_val_dataset, 'eval_test' : eval_test_dataset}
