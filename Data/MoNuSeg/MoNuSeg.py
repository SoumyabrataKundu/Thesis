import torch
from torchvision import transforms
import os
import tifffile as tiff
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import sys

sys.path.append('../../Steerable/')
import Steerable.utils

#####################################################################################################
##################################### MoNuSeg Dataset ###############################################
##################################################################################################### 

class AnnotationsToTensor:
    def __init__(self, target_shape):

        self.target_shape = target_shape

    def __call__(self, xml_file):
        
        polygons = self.parse_xml_to_polygons(xml_file)
        mask = self.create_mask_from_polygons(polygons, self.target_shape)
        
        return torch.from_numpy(mask)

    
    def parse_xml_to_polygons(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        polygons = []
        for region in root.findall(".//Region"):
            polygon = []
            for vertex in region.findall(".//Vertex"):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                polygon.append((x, y))
            polygons.append(polygon)
        
        return polygons

    def create_mask_from_polygons(self, polygons, image_size):
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        for polygon in polygons:
            draw.polygon(polygon, outline=1, fill=1)
        
        return np.array(mask)
    


class MoNuSeg(torch.utils.data.Dataset):
    def __init__(self, data_path, mode='train', image_transform = None, target_transform = None) -> None:

        self.image_transform = image_transform
        self.target_transform = target_transform
        
        self.data_path = data_path
        image_paths = os.path.join(data_path, f'MoNuSeg{mode.capitalize()}Data', 'Tissue Images')
        target_paths = os.path.join(data_path, f'MoNuSeg{mode.capitalize()}Data', 'Annotations')
        
        self.image_files = [os.path.join(image_paths, image) for image in os.listdir(image_paths)]
        self.target_files = [os.path.join(target_paths, target) for target in os.listdir(target_paths)]           
        
        self.n_samples = len(self.image_files)

    def __getitem__(self, index):
        image = tiff.imread(self.image_files[index])
        target = self.target_files[index]
        
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return self.n_samples
   
#####################################################################################################
######################################## Main Function ##############################################
##################################################################################################### 

URL = 'https://drive.usercontent.google.com/download?id=1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA&export=download&authuser=0'
def main(data_path):
    if data_path is None:
        Steerable.utils.download_and_unzip(URL, 'MoNuSeg')
        data_path = 'MoNuSeg/MoNuSeg/'

    image_shape = (1000, 1000)
    image_transform = transforms.ToTensor()
    target_transform = AnnotationsToTensor(image_shape)
    
    train_dataset = MoNuSeg(data_path, 'train', image_transform=image_transform, target_transform=target_transform)
    test_dataset = MoNuSeg(data_path, 'test', image_transform=image_transform, target_transform=target_transform)
    
    datasets =  {'train' : train_dataset, 'val': None, 'test' : test_dataset}

    hdf5file = Steerable.utils.HDF5Dataset('MoNuSeg.hdf5')
    for mode in datasets:
        hdf5file.create_hdf5_dataset(mode, datasets[mode])

    return
    
if __name__== '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/')
    args = parser.parse_args()

    main(**args.__dict__)