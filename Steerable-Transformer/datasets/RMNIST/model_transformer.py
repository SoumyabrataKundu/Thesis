import sys
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, n_layers = 1):
        super(TransformerEncoder, self).__init__()
        encoderlayer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoderlayer, n_layers)
    def forward(self, src):
        src = src.flatten(-2).transpose(-2,-1)
        return self.encoder(src)



class Model(nn.Module):
    def __init__(self, n_radius, max_m) -> None:
        super(Model, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.network = nn.Sequential(
            nn.Conv2d(1,  64, 5, padding = 'same'), # 28 X 28
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride = 4),     # 7 X 7
            nn.BatchNorm2d(128),
            

            TransformerEncoder(128,8,256, n_layers = 4),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7,10)
        )



    def forward(self, x):
        return self.network(x.type(torch.float))



import torch
import os
import torchvision
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
    #train_dataset = torchvision.datasets.MNIST(data_path, train=False, transform=transformations)

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset} 
