import os
import h5py
import torch



class HDF5Dataset:
    def __init__(self, filename : str, overwrite=False) -> None:
        self.filename = filename
        self.overwrite = overwrite
        
        self._create_hdf5_file()
        

    def _create_hdf5_file(self):
        if not os.path.isfile(self.filename) or self.overwrite:
            f = h5py.File(self.filename, 'w')
            f.close()
            
        else:
            raise FileExistsError("File Already Exists! To overwrite it, set overwrite=True")
                
        return
    
    def _initialize_hdf5_dataset(self, name, input_shape, target_shape, input_dtype, target_dtype):
        print(f'Creating {name} dataset: ', end='')
        self.file.create_dataset(name + '_inputs', (0, ) + input_shape, maxshape=(None,) +  input_shape, chunks=True, dtype=input_dtype)
        self.file.create_dataset(name + '_targets', (0,) + target_shape, maxshape=(None,) +  target_shape, chunks=True, dtype=target_dtype)
        print('Success!')

        return

    def create_hdf5_dataset(self, name, dataset, batched=False, variable_length=False):
        self.file = h5py.File(self.filename, 'a')
        input, target = dataset[0]
        target = target if torch.is_tensor(target) else torch.tensor(target)
        
        input_shape = input.shape[1:] if batched else input.shape
        target_shape = target.shape[1:] if batched else target.shape
        
        input_shape = () if variable_length else input_shape
        target_shape = () if variable_length else target_shape
        
        input_data_type = h5py.special_dtype(vlen=input.numpy().dtype) if variable_length else input.numpy().dtype
        target_dtype = h5py.special_dtype(vlen=target.numpy().dtype) if variable_length else target.numpy().dtype
        
        self._initialize_hdf5_dataset(name, input_shape, target_shape, input_data_type, target_dtype)
        
        for index in range(len(dataset)):
            try:
                input, target = dataset[index]
                target = target if torch.is_tensor(target) else torch.tensor(target)
                self._write_into_hdf5_file(name, input, target, batched, variable_length)
            except Exception as e:
                print(f'Excpetion at {index + 1} : {e}')
                
            print(f"Writing into {name} dataset : {index+1} / {len(dataset)}", end="\r")
        print()
        print('Done')
        self.file.close()
        
        return

    def _write_into_hdf5_file(self, name, input, target, batched, variable_length):
        inputs = self.file[name + '_inputs']
        targets = self.file[name + '_targets']
        
        input = input if batched else input.unsqueeze(0)
        target = target if batched else target.unsqueeze(0)
        assert input.shape[0] == target.shape[0], 'Number of examples in input and target should match.'
        
        input = input.flatten(1) if variable_length else input
        target = target.flatten(1) if variable_length else target
        
        inputs.resize((len(inputs) + len(input),) + tuple(inputs.shape[1:]))
        targets.resize((len(targets) + len(target),) + tuple(targets.shape[1:]))

        inputs[-len(input):] = input
        targets[-len(target):] = target
            
        
        return 
    
    
class HDF5(torch.utils.data.Dataset):
    def __init__(self, file, mode = 'train', image_transform = None, target_transform = None) -> None:

        if not mode in ["train", "test", "val"]:
            raise ValueError("Invalid mode")

        self.mode = mode
        self.file = file
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        # Reading from file
        input = torch.from_numpy(self.file[self.mode + '_inputs'][index]).float()
        target = torch.tensor(self.file[self.mode + '_targets'][index]).long()

        # Applying trasnformations
        if self.image_transform is not None:
            input = self.image_transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.file[self.mode+'_targets'])
