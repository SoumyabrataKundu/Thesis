import os
import h5py

import torch

from Steerable.utils import HDF5, RandomRotate

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.num_classes = 4
        transformer_dim = 208

        self.convolution_stem1 = torch.nn.Sequential(
            torch.nn.Conv3d(4,32,7, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(32,32,5, padding='same'),
            torch.nn.BatchNorm3d(32),
        )

        self.pool1 = torch.nn.AvgPool3d(4)

        self.convolution_stem2 = torch.nn.Sequential(
            torch.nn.Conv3d(32,64,5, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv3d(64, transformer_dim,5, padding='same'),
            torch.nn.BatchNorm3d(transformer_dim),
        )

        self.pool2 = torch.nn.AvgPool3d(4)

        self.encoder_decoder = torch.nn.Sequential(
            torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=4, batch_first=True), num_layers=4),
        ) 

        self.encoder_decoder = torch.nn.Sequential(
            torch.nn.Linear(transformer_dim, 2*transformer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*transformer_dim, transformer_dim),
        )
        self.norm = torch.nn.BatchNorm3d(transformer_dim)

        self.convolution_head1 = torch.nn.Sequential(
            torch.nn.Conv3d(2*transformer_dim,64,5, padding = 'same'),
            torch.nn.ReLU(),
            torch.nn.Conv3d(64,32,5, padding = 'same'),
            torch.nn.BatchNorm3d(32),
        )

        self.convolution_head2 = torch.nn.Sequential(
            torch.nn.Conv3d(2*32,16,5, padding = 'same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(16),
            torch.nn.Conv3d(16,self.num_classes,5, padding = 'same'),
        )
        
    def forward(self, x):
        full_shape = x.shape

        # Downsampling
        ## Downsampling I
        stem1 = self.convolution_stem1(x)
        x = self.pool1(stem1)
        ## Downsampling II
        stem2 = self.convolution_stem2(x)
        x = self.pool2(stem2)

        # Encoder-Decoder
        x_shape = x.shape
        x = self.encoder_decoder(x.flatten(2).transpose(1,2))
        x = x.transpose(1,2).reshape(*x_shape)

        # Upsampling
        ## Upsampling I
        x = torch.nn.functional.interpolate(x, size=stem2.shape[-3:], mode="trilinear")
        x = self.convolution_head1(torch.cat([x, stem2], dim=1)) # skip connection
        ## Upsampling II
        x = torch.nn.functional.interpolate(x, size=stem1.shape[-3:], mode="trilinear")
        x = self.convolution_head2(torch.cat([x, stem1], dim=1)) # skip connection
        ## Upsampling II
        x = torch.nn.functional.interpolate(x, size=full_shape[-3:], mode="trilinear")

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
