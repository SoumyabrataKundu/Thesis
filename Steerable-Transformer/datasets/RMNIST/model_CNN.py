import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_radius, max_m):
        super(Model, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)  # Output size is 10 for 10 classes (0-9)

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = x.type(torch.float)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the feature maps
        x = x.view(-1, 128 * 3 * 3)
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




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
    train_dataset = torchvision.datasets.MNIST(data_path, train=False, transform=transformations)

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset} 
