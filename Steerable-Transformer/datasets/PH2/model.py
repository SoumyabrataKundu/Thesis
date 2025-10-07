import torch
import torchvision


#from SteerableSegmenter2D.conv_layers import *
#from SteerableSegmenter2D.transformer_layers import SE2TransformerEncoder, SE2TransformerDecoder, SE2ClassEmbedings, SE2LinearDecoder

from Steerable.nn import *

class Model(torch.nn.Module):
    def __init__(self, n_radius, max_m) -> None:
        super(Model, self).__init__()
        
        device='cuda' 
        n_theta = 40
        self.num_classes = 2
        encoder_dim = 64
        decoder_dim = 64

        self.convolution_stem1 = nn.Sequential(
            SE2ConvType1(3,8,5, n_radius, n_theta, max_m, padding='same'),
            SE2NormNonLinearity(8, max_m),
            SE2ConvType2(8,16,5, n_radius, n_theta, max_m, padding='same'),
            SE2BatchNorm(),
        )
        
        self.pool1 = SE2AvgPool(8)
  
        self.convolution_stem2 =  nn.Sequential(
            SE2ConvType2(16,32,5, n_radius, n_theta, max_m, padding='same'),
            SE2NormNonLinearity(32, max_m),
            SE2ConvType2(32,encoder_dim,5, n_radius, n_theta, max_m, padding='same'),
            SE2BatchNorm(),

        )

        self.pool2 = SE2AvgPool(8)

        self.encoder = SE2TransformerEncoder(encoder_dim, 8, max_m, n_layers=2, add_pos_enc=False)
        #self.linear_decoder = SE2LinearDecoder(encoder_dim, decoder_dim, max_m)
        #self.decoder = SE2TransformerDecoder(encoder_dim, 8, max_m, self.num_classes, n_layers=1, add_pos_enc=True)
        
        self.convolution_head1 = nn.Sequential(
            SE2ConvType2(decoder_dim,32,5, n_radius, n_theta, max_m, padding = 'same'),
            SE2NormNonLinearity(32, max_m),
            SE2ConvType2(32,16,5, n_radius, n_theta, max_m, padding = 'same'),
            SE2BatchNorm(),
        )

        self.convolution_head2 = nn.Sequential(
            SE2ConvType2(16,8,5, n_radius, n_theta, max_m, padding = 'same'),
            SE2NormNonLinearity(8, max_m),
            SE2BatchNorm(),
            SE2ConvType2(8, 8,5, n_radius, n_theta, max_m, padding = 'same'),
            #SE2BatchNorm(),
        )

        #self.embed = SE2ClassEmbeddings(encoder_dim, 8, max_m)
       
        
        
    def forward(self, x):
        x_shape = x.shape
        x = x.type(torch.cfloat)
        
        # Downsampling
        stem1 = self.convolution_stem1(x)
        x = self.pool1(stem1)
        stem2 = self.convolution_stem2(x)
        x = self.pool2(stem2)

        # Encoder
        x = self.encoder(x)

        # Decoder
        #x, classes = self.decoder(x)
        #x = self.linear_decoder(x)

        # Upsampling
        x = nn.functional.interpolate(x.real, size=(x.shape[-3], *stem2.shape[-2:]), mode="trilinear") + \
                  1j * nn.functional.interpolate(x.imag, size=(x.shape[-3], *stem2.shape[-2:]), mode="trilinear")
        x = self.convolution_head1(x + stem2) # skip connection

        x = nn.functional.interpolate(x.real, size=(x.shape[-3], *stem1.shape[-2:]), mode="trilinear") + \
                  1j * nn.functional.interpolate(x.imag, size=(x.shape[-3], *stem1.shape[-2:]), mode="trilinear")
        x = self.convolution_head2(x + stem1) # skip connection
 
        # Norm
        #x = self.embed(x, classes).abs()
        x = torch.linalg.vector_norm(x, dim=1)
        #x, _ = torch.max(x.abs(), dim=1)
    
        return x

#######################################################################################################################
###################################################### Dataset ########################################################
####################################################################################################################### 



import torch
import os
import torchvision.transforms as transforms
import h5py

class PH2(torch.utils.data.Dataset):
    def __init__(self, file, mode = 'train', image_transform = None, target_transform = None) -> None:

        if not mode in ["train", "test"]:
            raise ValueError("Invalid mode")

        self.mode = mode
        self.file = file
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.n_samples = len(self.file[mode+'_targets'])

    def __getitem__(self, index):

        # Reading from file
        img = torch.from_numpy(self.file[self.mode + '_images'][index])
        target = self.file[self.mode + '_targets'][index]

        # Applying trasnformations
        if self.image_transform is not None:
            img = self.image_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, torch.from_numpy(target).long()

    def __len__(self):
        return self.n_samples


def get_datasets(data_path):
    # Load the dataset
    data_file = h5py.File(os.path.join(data_path, 'PH2.hdf5'), 'r')

    # Transformations
    image_transform = transforms.Compose([
        transforms.Normalize(mean=0, std = 1)
        ])

    # Load datasets
    train_dataset = PH2(data_file, mode='train', image_transform=image_transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [100, 50])
    test_dataset = PH2(data_file, mode='test', image_transform=image_transform)


    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}
