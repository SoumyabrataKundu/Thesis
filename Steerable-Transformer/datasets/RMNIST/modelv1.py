import sys
import torch
import torchvision
import torchvision.transforms as transforms

from SteerableTransformer2D.conv_layers import *
from SteerableTransformer2D.transformer_layers import *


class Model(nn.Module):
    def __init__(self, n_radius, max_m) -> None:
        super(Model, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_theta = 40

        self.network = nn.Sequential(
            FintConv2DType1(1,  4, 5, n_radius, n_theta, max_m, padding='same', device = device), # 28 X 28
            CGNonLinearity2D(max_m, device = device),
            FintConv2DType2(4,  4, 5, n_radius, n_theta, max_m, padding='same', device = device), # 28 X 28

            SteerableBatchNorm2D(),

            FintAvgPool2D(2),                                                                      # 14 X 14
            FintConv2DType2(4, 8, 5, n_radius, n_theta, max_m, padding='same', device = device), # 14 X 14
            CGNonLinearity2D(max_m, device = device),
            FintConv2DType2(8, 8, 5, n_radius, n_theta, max_m, padding='same', device = device), # 14 X 14

            SteerableBatchNorm2D(),

            FintAvgPool2D(2),

            #SE2TransformerEncoder(8,2,max_m, n_layers = 4,device = device),

            FintConv2DType2(8, 8, 5, n_radius, n_theta, max_m, padding='same', device = device), #  7 X  7
            CGNonLinearity2D(max_m, device = device),
            FintConv2DType2(8, 16, 5, n_radius, n_theta, max_m, padding='same', device = device), #  7 X  7

            SteerableBatchNorm2D(),

            FintConv2DType2(16, 32, 5, n_radius, n_theta, max_m, padding='same', device = device), #  7 X  7
         
            SE2TransformerEncoder(32,4,max_m, n_layers = 4,device = device),

            NormFlatten(),
            torch.nn.Linear(32,10),
          )


    def forward(self, x):
        return self.network(x)
  





import torch
import os
import torchvision.transforms as transforms
import h5py

class RotMNIST(torch.utils.data.Dataset):
    def __init__(self, file, mode = 'train', image_transform = None, target_transform = None) -> None:
        
        if not mode in ["train", "test", "val"]:
            raise ValueError("Invalid mode")
        
        self.mode = mode
        self.file = file
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.n_samples = len(self.file[mode+'_targets'])

    def __getitem__(self, index):
        
        # Reading from file
        img = torch.from_numpy(self.file[self.mode + '_images'][index]).unsqueeze(0)
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
    data_file = h5py.File(os.path.join(data_path, 'rotated_mnist.hdf5'), 'r')
    
    # Transformations
    image_transform = transforms.Compose([
        transforms.Normalize(mean=0, std = 1) 
        ])
    
    # Load datasets
    train_dataset = RotMNIST(data_file, mode='train', image_transform=image_transform)
    val_dataset = RotMNIST(data_file, mode='val', image_transform=image_transform)
    test_dataset = RotMNIST(data_file, mode='test', image_transform=image_transform)
    
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    test_set_size = int(len(test_dataset) * 0.95)
    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [test_set_size, len(test_dataset) - test_set_size])
    
    transformations = transforms.Compose([
        #transforms.RandomRotation(degrees=(0, 360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std = 1)
        ])
    #test_dataset = torchvision.datasets.MNIST(data_path, train=True, transform=transformations)

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset} 
