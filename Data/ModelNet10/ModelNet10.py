import torch
import numpy as np
import h5py
import os
import fnmatch
import sys

sys.path.append('../../Steerable/')
import Steerable.utils

# Dataset Generation
class ModelNet10(torch.utils.data.Dataset):
    def __init__(self, data_path, size, mode='train', rotate=False, rotate_z=False, jitter=False) -> None:
        if mode not in ['train', 'test']:
            raise ValueError(f'Invalid mode {mode}. Should be one of train or test.')
        
        self.size = size
        self.rotate = rotate
        self.rotate_z = rotate_z
        self.jitter = jitter
        self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if fnmatch.fnmatch(f, mode+'*.h5')]
        self.length = []

        for file in self.files:
            with h5py.File(file, 'r') as f:
                self.length.append(len(f['label']))

    def __getitem__(self, index):
        indices, label = self.get_indices(index)
        if self.rotate:
            indices = self.rotate_point_cloud_3d(indices)
        elif self.rotate_z:
            indices = self.rotate_point_cloud_3d_z(indices)
        if self.jitter:
            indices = self.jitter_point_cloud(indices)

        indices = ((indices + 1) * self.size/2).astype(int)
        image = torch.zeros(1, *[self.size]*3)
        for x_value, y_value, z_value in indices:
            image[0, x_value, y_value, z_value] = 1
            
        return image, torch.tensor(label)


    def get_indices(self, index):            
        running_sum=0
        for file, length in zip(self.files, self.length):
            if index < running_sum + length:
                with h5py.File(file, 'r+') as f:
                    return f['data'][index - running_sum], f['label'][index - running_sum].item()
            running_sum += length
            
        raise IndexError('Index Out of Bounds.')

    def __len__(self):
        return sum(self.length)
        
    def rotate_point_cloud_3d(self, indices):
        # uniform sampling
        angles = np.random.uniform(size = [3]) * np.pi * np.array([2,1,2])

        Rz1 = np.array([[np.cos(angles[0]), -np.sin(angles[0]), 0],
                        [np.sin(angles[0]), np.cos(angles[0]), 0],
                        [0, 0, 1]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                        [0, 1, 0],
                        [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz2 = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                        [np.sin(angles[2]), np.cos(angles[2]), 0],
                        [0, 0, 1]])
        R = Rz2 @ Ry @ Rz1
        rotated_indices = np.dot(indices, R)

        return rotated_indices



    def rotate_point_cloud_3d_z(self, indices):
        # uniform sampling
        angle = np.random.uniform() * 2 * np.pi
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
        rotated_indices = np.dot(indices, R)

        return rotated_indices


    def jitter_point_cloud(self, indices, sigma=0.01, clip=0.05):
        N, C = indices.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        jittered_data = np.clip(jittered_data + indices, -1+0.005, 1-0.005)
        return jittered_data

def main(data_path, size, rotate, rotate_z, jitter):
    filename = 'ModelNet10_' + ('rotate' if rotate else ('rotate_z' if rotate_z else '')) + ('jitter' if jitter else '') + str(size) + '.hdf5'
    hdf5file = Steerable.utils.HDF5Dataset(filename)

    for mode in ['train', 'test']:
        dataset = ModelNet10(data_path=data_path, mode = mode, size=size, rotate=rotate, rotate_z=rotate_z, jitter=jitter)
        hdf5file.create_hdf5_dataset(mode, dataset)
        
if __name__== '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/modelnet10/')
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--rotate", type=bool, default=False)
    parser.add_argument("--rotate_z", type=bool, default=True)
    parser.add_argument("--jitter", type=bool, default=False)
    

    args = parser.parse_args()

    main(**args.__dict__)
