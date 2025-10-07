import torch

class AddGaussianNoise(torch.utils.data.Dataset):
    def __init__(self, dataset:torch.utils.data.Dataset, sd:float):
        self.dataset = dataset
        self.sd = sd
        
    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        inputs = inputs + torch.randn_like(inputs) * self.sd
        return inputs, targets
            
    def __len__(self):
        return len(self.dataset)