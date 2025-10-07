import torch
import h5py
import sys
sys.path.append('../../Steerable')

from Steerable.datasets.hdf5 import HDF5, HDF5Dataset
from Steerable.Segmentation.Patchify import PatchifyDataset

class image_reshape:
    def __call__(self, image):
        return image.reshape(1,512,512,-1)
    
class target_reshape:
    def __call__(self, image):
        return image.reshape(512,512,-1)
    

def main():
    task = 'Pancreas'
    train_dataset = HDF5(h5py.File(f'{task}/data/{task}.hdf5', 'r'), mode='train', image_transform=image_reshape(), target_transform=target_reshape())
    val_dataset = HDF5(h5py.File(f'{task}/data/{task}.hdf5', 'r'), mode='val', image_transform=image_reshape(), target_transform=target_reshape())
    test_dataset = HDF5(h5py.File(f'{task}/data/{task}.hdf5', 'r'), mode='test', image_transform=image_reshape(), target_transform=target_reshape())

    datasets = {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}

    hdf5file = HDF5Dataset(f'{task}_patched.hdf5')

    for mode in datasets:
        hdf5file.create_hdf5_dataset(mode, PatchifyDataset(datasets[mode], kernel_size=(128,128,64), stride=(128,128,64)), batched=True)

    return

if __name__ == "__main__":
    main()
