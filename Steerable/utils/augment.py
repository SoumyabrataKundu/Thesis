import torch


class Augment(torch.utils.data.Dataset):
    def __init__(self, dataset, function, parameters, batched=False):
        self.dataset = dataset
        self.function = function
        self.parameters = parameters
        self.batched = batched
        
    def __getitem__(self, index):
        if self.batched:
            data = self.dataset[index]
            aug_data = [self.function(data, self.parameters[j]) for j in range(len(self.parameters))]
            return torch.stack([aug[0] for aug in aug_data], dim=0), torch.stack([aug[1] for aug in aug_data], dim=0)
        else:
            i = index // len(self.parameters)
            j = index % len(self.parameters)
            return self.function(self.dataset[i], self.parameters[j])
            
        
    def __len__(self):
        return len(self.dataset) if self.batched else len(self.dataset) * len(self.parameters)