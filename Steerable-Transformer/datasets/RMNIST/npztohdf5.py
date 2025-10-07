import h5py
import os
import numpy as np
from math import sqrt

def main(data_path):
    # Loading the data and saving in hdf5 format
    with h5py.File(os.path.join(data_path, 'rotated_mnist_new.hdf5'), 'w') as f:
        size=28
        for mode in ['test', 'train']: ## Mode is organized in loop as increasing number of examples in each case.
            print(mode + '...', end = "")
            
            # Reading the npz file
            data = np.load(os.path.join(data_path, 'mnist_rotation_'+ mode + '.amat'))
            size = sqrt(data['x'].shape[-1])
            
            # Creating dataset
            f.create_dataset(mode + '_images', (len(data['x']), *[size]*2), chunks=True)
            f.create_dataset(mode + '_targets', (len(data['y']), ), maxshape=(None,), chunks=True)
            
            # Data subset loading for each mode
            images = f[mode + '_images']
            targets = f[mode + '_targets']
            
            images[:] = data['x'].reshape(-1, 28, 28)
            targets[:] = data['y']
            
            print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    main(**args.__dict__)


########################### Testing ######################################

# with h5py.File('../../../../data/RotatedMNIST/rotated_mnist.hdf5', 'r') as f:
#     for mode in ['val', 'test', 'train']:
#         mode = 'val'
#         # Reading the npz file
#         data = np.load(os.path.join('../../../../data/RotatedMNIST/', 'rotated_'+ mode + '.npz'))
        
#         images = f[mode + '_images']
#         targets = f[mode + '_targets']
        
#         print(np.all(images == data['x'].reshape(-1, 28, 28)))
#         print(np.all(targets == data['y']))
