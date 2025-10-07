import torch
import os
from PIL import Image
from torchvision import transforms
import sys

sys.path.append('../../Steerable/')
from Steerable.datasets.hdf5 import HDF5Dataset
    
    
    
#####################################################################################################
##################################### PH2 Dataset Class #############################################
##################################################################################################### 


def get_max_size(data_path):
    data_path = data_path + '/PH2 Dataset images'
    image_folders = os.listdir(data_path)
    transform = transforms.ToTensor()            
    size = []
    for folder in image_folders:
        image = Image.open(os.path.join(data_path, folder, folder+'_Dermoscopic_Image', folder+'.bmp'))
        target = Image.open(os.path.join(data_path, folder, folder+'_lesion', folder+'_lesion.bmp'))

        image_tensor = transform(image)
        target_tensor = transform(target)
        assert image_tensor.shape[1:] == target_tensor.shape[1:]
        size.append(list(image_tensor.shape))
        
    size_tensor = torch.tensor(size)
    return torch.max(size_tensor[:,1]).item(), torch.max(size_tensor[:,2]).item()


class PH2(torch.utils.data.Dataset):
    def __init__(self, data_path, image_transform = None, target_transform = None) -> None:

        self.image_transform = image_transform
        self.target_transform = target_transform
        
        self.data_path = data_path + '/PH2 Dataset images'
        self.image_folders = os.listdir(self.data_path)          
        
        self.n_samples = len(self.image_folders)

    def __getitem__(self, index):
        folder = self.image_folders[index]
        image = Image.open(os.path.join(self.data_path, folder, folder+'_Dermoscopic_Image', folder+'.bmp'))
        target = Image.open(os.path.join(self.data_path, folder, folder+'_lesion', folder+'_lesion.bmp'))
        
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target[0].long()

    def __len__(self):
        return self.n_samples

#####################################################################################################
######################################## Main Function ##############################################
##################################################################################################### 

def main(data_path):
    max_size = (578, 770) #get_max_size(data_path)
    transformation = transforms.Compose([
            transforms.Resize(max_size),
            transforms.ToTensor(),
            ])
    dataset = PH2(data_path, image_transform=transformation, target_transform=transformation)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.1, 0.2])
    
    datasets = {'train' : train_dataset, 'val' : val_dataset 'test' : test_dataset}
    hdf5file = HDF5Dataset('PH2.hdf5')
    for mode in datasets:
        hdf5file.create_hdf5_dataset(mode, datasets[mode])

    return
    
    
if __name__== '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/')

    args = parser.parse_args()
    main(**args.__dict__)
