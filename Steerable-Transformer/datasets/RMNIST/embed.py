import torch
import h5py
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms

def main(data_path, factor, size):
    #rot_mnist = h5py.File(os.path.join(data_path, 'rotated_mnist.hdf5'), 'r')
    transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0, std = 1)
            ])
    with h5py.File(os.path.join(data_path, 'mnist_embed_noise.hdf5'), 'w') as f:
        for mode in ['val', 'train', 'test']: ## Mode is organized in loop as increasing number of examples in each case.
            print(mode + '...')

            rot_mnist_images = torchvision.datasets.MNIST("./data", train = mode == "train", transform=transformations)
            #rot_mnist_images = rot_mnist[mode + '_images']
            #rot_mnist_targets = rot_mnist[mode + '_targets']

            # Creating dataset
            f.create_dataset(mode + '_images', (len(rot_mnist_images), *[factor * size]*2), chunks=True)
            f.create_dataset(mode + '_targets', (len(rot_mnist_images), ), chunks=True)

            embed_images = f[mode + '_images']
            embed_targets = f[mode + '_targets']

            for index in range(len(rot_mnist_images)):
                # Data subset loading for each mode
                #image = rot_mnist_images[index]
                #target = rot_mnist_targets[index]
                image, target = rot_mnist_images[index]
                image = image[0]
 
                # Embedding
                height, width = image.shape[1], image.shape[0]
                embedded_image = np.zeros((factor*height,factor*width))
                x = torch.randint(0, (factor-1)*width,(1,)).item()
                y = torch.randint(0, (factor-1)*height,(1,)).item()
                embedded_image[x:(x+width), y:(y+height)] = image
                embedded_image = embedded_image + np.random.randn(size * factor, size * factor) * 0.1
                
                # Saving in hdf5
                embed_images[index, :, :] = embedded_image
                embed_targets[index] = target
                
                print(f"{index} / {len(rot_mnist_images)}", end = "\r")

            print("Done!")
            
            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)

    args = parser.parse_args()

    main(**args.__dict__)
