import sys
#from SteerableTransformer3D.transformer_layers import *
#from SteerableTransformer3D.conv_layers import *
import torch
import torch.nn as nn

############################################################################################################################
###################################################### Model ###############################################################
############################################################################################################################

class Model(nn.Module):
    def __init__(self, n_radius, maxl) -> None:
        super(Model, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_angle = 40

        self.network = nn.Sequential(
            FintConv3DType1(1, 9, 5, n_radius, n_angle, maxl, padding='same', device = device),     # 32 X 32 X 32
            CGNonLinearity3D(9, 9, maxl),
            FintConv3DType2(9, 18, 3, n_radius, n_angle, maxl, padding='same', device = device),    # 32 X 32 X 32
            SteerableBatchNorm3D(),

            FintAvgPool3D(4),

            SE3TransformerEncoder(18, 6, maxl, n_layers = 1, device=device),
         
            #FintConv3DType2(18, 18, 3, n_radius, n_angle, maxl, padding='same', device = device),   # 16 X 16 X 16
            #CGNonLinearity3D(18, 18, maxl),
            #FintConv3DType2(18, 36, 3, n_radius, n_angle, maxl, padding='same', device = device),   # 16 X 16 X 16

            #SteerableBatchNorm3D(),

            #FintAvgPool3D(2),

            FintConv3DType2(18, 36, 3, n_radius, n_angle, maxl, padding='same', device = device),   #  8 X  8 X  8
            CGNonLinearity3D(36, 36, maxl),
            FintConv3DType2(36, 72, 3, n_radius, n_angle, maxl, padding='same', device = device),   #  8 X  8 X  8
            SteerableBatchNorm3D(),

            FintAvgPool3D(4),

            #FintConv3DType2(72, 72, 3, n_radius, n_angle, maxl, padding='same', device = device),   #  4 X  4 X  4
            #CGNonLinearity3D(72, 72, maxl),
            #FintConv3DType2(72, 144, 3, n_radius, n_angle, maxl, padding='same', device = device),  #  4 X  4 X  4

            #SteerableBatchNorm3D(),

            SE3TransformerEncoder(72, 8, maxl, n_layers = 1, device=device),
            #SteerableBatchNorm3D(),

            FintConv3DType2(72, 144, 2, n_radius, n_angle, maxl, device = device), #  4 X  4 X  4

            NormFlatten(),
            #torch.nn.Linear(144, 10),
            torch.nn.Linear(144, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


from math import sqrt
class Model(nn.Module):
    def __init__(self, n_radius, maxl) -> None:
        super(Model, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_angle = 40
        maxl = int(sqrt(maxl))

        self.conv1 = nn.Sequential(
            torch.nn.Conv3d(1, 9*maxl, 5, padding='same'),     # 32 X 32 X 32
            torch.nn.ReLU(),
            torch.nn.Conv3d(9*maxl, 36*maxl, 3, padding='same'),    # 32 X 32 X 32
            torch.nn.BatchNorm3d(36*maxl),

            torch.nn.AvgPool3d(4),
            )

        #self.encoder1 = torch.nn.TransformerEncoderLayer(18*maxl, 6*maxl, 18*maxl, batch_first=True)

        self.conv2 = nn.Sequential(
            torch.nn.Conv3d(36*maxl, 36*maxl, 3, padding='same'),   #  8 X  8 X  8
            torch.nn.ReLU(),
            torch.nn.Conv3d(36*maxl, 72*maxl, 3, padding='same'),   #  8 X  8 X  8
            torch.nn.BatchNorm3d(72*maxl),

            torch.nn.AvgPool3d(4),
            )
        #self.encoder2 = torch.nn.TransformerEncoderLayer(72*maxl, 8*maxl, 72*maxl, batch_first=True)

        self.conv3 = nn.Sequential(
            torch.nn.Conv3d(72*maxl, 144*maxl, 2), #  4 X  4 X  4
            torch.nn.Flatten(),
            torch.nn.Linear(144*maxl, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        #x = self.encoder1(x.flatten(2).transpose(1,2)).transpose(1,2)
        x = self.conv2(x.reshape(x.shape[0], x.shape[1], 8,8,8))
        #x = self.encoder2(x.flatten(2).transpose(1,2)).transpose(1,2)
        x = self.conv3(x.reshape(x.shape[0], x.shape[1], 2,2,2))
        return x

##############################################################################################################################
################################################# ModelNet10 Dataset ############################################################
##############################################################################################################################

import torch
import torch.utils.data
import h5py
import os

class ModelNet10(torch.utils.data.Dataset):
    def __init__(self, file, mode = 'train', image_transform = None, target_transform = None) -> None:

        if not mode in ["train", "test"]:
            raise ValueError("Invalid mode")

        self.mode = mode
        self.file = file
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.n_samples = len(self.file[mode+'_targets'])

    def __getitem__(self, index):

        # Reading from file
        img = self.file[self.mode + '_images'][index]
        target = self.file[self.mode + '_targets'][index]

        # Applying trasnformations
        if self.image_transform is not None:
            img = self.image_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.n_samples

def get_datasets(data_path):
    # Load the dataset
    data_file = h5py.File(os.path.join(data_path, 'modelnet10_transformed.hdf5'), 'r')

    # Transformations
    def image_transform(x):
        x = (torch.from_numpy(x).unsqueeze(0) * 6) -1
        return x.float()

    # Load datasets
    train_dataset = ModelNet10(data_file, mode='train', image_transform=image_transform, target_transform = None)
    test_dataset = ModelNet10(data_file, mode='test', image_transform=image_transform, target_transform = None)
    val_dataset = ModelNet10(data_file, mode='test', image_transform=image_transform, target_transform = None)
    #test_set_size = int(len(test_dataset) * 0.90)
    #test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [test_set_size, len(test_dataset) - test_set_size]) 

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}



