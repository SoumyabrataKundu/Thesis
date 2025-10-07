import torch
import torchvision
import sys
import torchvision.transforms as transforms

sys.path.append('../../Steerable/')
from Steerable.datasets import HDF5Dataset


def get_datasets(data_path):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std = 1)
        ])

    # Load datasets
    full_train_dataset = torchvision.datasets.MNIST(data_path, train=True, transform=transformations)
    full_test_dataset = torchvision.datasets.MNIST(data_path, train=False, transform=transformations)

    test_dataset, partial_dataset = torch.utils.data.random_split(full_train_dataset, [len(full_train_dataset)-2000, 2000]) 
    train_dataset = torch.utils.data.ConcatDataset([full_test_dataset, partial_dataset])

    return { 'train' : train_dataset, 'test' : test_dataset }

def main(data_path):
    filename = 'MNIST.hdf5'
    hdf5file = HDF5Dataset(filename)
    datasets = get_datasets(data_path)

    for mode in datasets:
        hdf5file.create_hdf5_dataset(mode, datasets[mode])

if __name__== '__main__':


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./')

    args = parser.parse_args()

    main(**args.__dict__)
