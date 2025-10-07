import torch
from scipy.ndimage.interpolation import rotate

def rotate_image2D(image, degree, order=1):
    assert 0 <= order <= 5, "'order' takes integer values between 0 and 5."
    assert image.ndim >= 2, "number of dimensions shoulg be atleast 2."

    image_shape = image.shape
    image = image.reshape(-1, *image_shape[-2:])
    
    image = torch.vstack([torch.from_numpy(rotate(image[i], degree, (1,0), reshape=False, order=order))
                          for i in range(image.shape[0])]).view(*image_shape)
    
    return image


def rotate_image3D(image, degree, order=1):
    assert 0 <= order <= 5, "'order' takes integer values between 0 and 5."
    assert image.ndim >= 3, "number of dimensions shoulg be atleast 2."

    def rotate_slice_image(image_slice, degree):
        image_slice = torch.from_numpy(rotate(image_slice, degree[0], (1,0), reshape=False, order=order))
        image_slice = torch.from_numpy(rotate(image_slice, degree[1], (0,2), reshape=False, order=order))
        image_slice = torch.from_numpy(rotate(image_slice, degree[2], (1,0), reshape=False, order=order))
        
        return image_slice

    image_shape = image.shape
    image = image.reshape(-1, *image_shape[-3:])
    
    image = torch.vstack([rotate_slice_image(image[i], degree) for i in range(image.shape[0])]).view(*image_shape)
    
    return image


class RandomRotate(torch.utils.data.Dataset):
    def __init__(self, dataset:torch.utils.data.Dataset, order=1):
        self.dataset = dataset
        self.order = order
        
    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        if inputs.ndim == 3:
            degree = torch.randint(0, 360, (1,)).item()
            inputs = rotate_image2D(inputs, degree=degree, order=self.order)
            if targets.ndim == 2:
                targets = rotate_image2D(targets, degree, order=0)
        elif inputs.ndim == 4:
            degree = torch.randint(0, 360, (3,))
            inputs = rotate_image3D(inputs, degree, order=self.order)
            if targets.ndim == 3:
                targets = rotate_image3D(targets, degree, order=0)
        else:
            ValueError("Only 2D or 3D image data are supported.")
        return inputs, targets
 
    def __len__(self):
        return len(self.dataset)
