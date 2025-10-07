import torch
import torchvision
from scipy.ndimage.interpolation import rotate
import sys

sys.path.append('../../Steerable/')
import Steerable.utils

class RotFMNIST(torch.utils.data.Dataset):
    def __init__(self, data_path, mode, rotate=False) -> None:
        if mode not in ['train', 'test']:
            raise ValueError(f'Invalid mode {mode}. Should be one of train or test.')

        self.rotate = rotate

        transformations = torchvision.transforms.ToTensor()

        full_test_dataset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transformations)
        full_train_dataset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transformations)

        test_dataset, partial_dataset = torch.utils.data.random_split(full_test_dataset, [58000, 2000])
        train_dataset = torch.utils.data.ConcatDataset([full_train_dataset, partial_dataset])
        if mode == 'train':
            self.data = train_dataset
        if mode == 'test':
            self.data = test_dataset

    def __getitem__(self, index):
        image, label = self.data[index] 

        if self.rotate:
            image = rotate(image[0], torch.randint(0, 360, (1,)).item(), reshape=False, order = 1)
            image = torch.from_numpy(image).reshape(*image.shape) 

        return image, label

    def __len__(self):
        return len(self.data)

def main(data_path, rotate):
    if data_path is None:
        data_path = './data/'
        torchvision.datasets.FashionMNIST(data_path, download=True)

    filename = ('Rot' if rotate else '') + 'FMNIST.hdf5'
    hdf5file = Steerable.utils.HDF5Dataset(filename)

    for mode in ['train', 'test']:
        hdf5file.create_hdf5_dataset(mode, RotFMNIST(data_path=data_path, mode=mode, rotate=rotate))
    
    
if __name__== '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--rotate", type=str, default=False)

    args = parser.parse_args()
    main(**args.__dict__)