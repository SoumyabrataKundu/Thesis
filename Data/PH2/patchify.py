import torch
import h5py
import sys

sys.path.append('../../Steerable')
from Steerable.utils import HDF5, HDF5Dataset, PatchifyDataset 

def main():
    data_file = h5py.File('data/PH2.hdf5', 'r')
    datasets = {'train' : HDF5(data_file, mode='train'), 'val': HDF5(data_file, mode='val'), 'test' : HDF5(data_file, mode='test')}
    hdf5file = HDF5Dataset('PH2_patched256_128.hdf5')
    for mode in ['train', 'val', 'test']:
        hdf5file.create_hdf5_dataset(mode, PatchifyDataset(HDF5(data_file, mode=mode), kernel_size=(256,256), stride=128), batched=True)

    return

if __name__ == "__main__":
    main()
