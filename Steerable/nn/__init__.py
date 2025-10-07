import importlib

from Steerable.nn.utils import *
from Steerable.nn.Steerable2d.conv_layers import *
from Steerable.nn.Steerable2d.transformer_layers import *
from Steerable.nn.Steerable3d.Steerable3d_pytorch.conv_layers import *
from Steerable.nn.Steerable3d.Steerable3d_pytorch.transformer_layers import *

backend = 'Pytorch'
def set_backend(set_to_backend : str):
    global backend
    
    if backend == set_to_backend:
        print(f'Already at Backend {backend}')
        return
    if set_to_backend == 'Pytorch':
        module = importlib.import_module('Steerable.nn.Steerable3d.Steerable3d_pytorch.conv_layers')

    elif set_to_backend == 'GElib':
        if importlib.util.find_spec('gelib') is None:
            print('GElib is not installed. Reverting to Pytorch')
            return
        else:
            module = importlib.import_module('Steerable.nn.Steerable3d.Steerable3d_gelib.conv_layers')
        
    else:
        raise ValueError(f"Incorrect backend {set_to_backend}. Should be either Pytorch or GElib.")
        
    backend = set_to_backend
    print(f'Set to Backend {set_to_backend}')
    globals().update({name: getattr(module, name) for name in dir(module) if not name.startswith('_')})
