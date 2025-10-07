from math import sqrt
from scipy.ndimage import map_coordinates
from scipy.special import sph_harm
from sympy.physics.quantum.cg import CG

import torch

###########################################################################################################################
############################################# Interpolation Matrix ########################################################
###########################################################################################################################

def get_interpolation_matrix(kernel_size, n_radius, n_angle, interpolation_order=1):
    assert 0 <= interpolation_order <= 5, "'interpolation_order' takes integer values between 0 and 5."
    d = len(kernel_size)
    assert d>=2, "dimension of 'kernel_size' should be atleast 2"
    R = torch.tensor([(kernel_size[i] - 1)/2 for i in range(d)])
    A1 = torch.pi * (torch.arange(n_angle)+0.5) / n_angle
    A2 = 2 * torch.pi * torch.arange(n_angle) / n_angle
    sphere_coord = torch.ones(1)
    r_values = torch.vstack([torch.arange(1, n_radius+1)*h/n_radius for h in R])
    for i in range(d-1):
        A = A1 if i<d-2 else A2
        sphere_coord = torch.vstack([
                    torch.tensordot(sphere_coord[:-1], torch.ones(n_angle), dims=0),
                    torch.tensordot(sphere_coord[-1:], torch.cos(A), dims=0), 
                    torch.tensordot(sphere_coord[-1:], torch.sin(A), dims=0)])
    sphere_coord = (torch.einsum('dr, da -> dra', r_values, sphere_coord.flatten(1)) + R.reshape(-1, 1,1)).flatten(1)
    
    if interpolation_order == 0:
        I = torch.zeros(n_radius * n_angle**(d-1), *[kernel_size[i]+1 for i in range(d)], dtype=torch.float)
        sphere_coord = sphere_coord.T.unsqueeze(1)
        integer = torch.floor(sphere_coord).type(torch.int64)
        fraction = sphere_coord - integer
        increment = torch.cartesian_prod(*[torch.tensor([0,1]) for _ in range(d)]).unsqueeze(0)
        values = ((((1 - fraction) ** (1 - increment)) * (fraction ** increment)) >= 0.5 - 1e-7).all(dim=2).flatten().float()
        coords = torch.cat([torch.arange(I.shape[0]).repeat_interleave(2**d).unsqueeze(1), (integer + increment).flatten(0,1)], dim=1)
        
        I[tuple(coords.T.tolist())] = values
        I = I[(Ellipsis,) + (slice(0,-1),)*d]
        I = I / I.sum(dim=torch.arange(1,d+1).tolist(), keepdim=True)
    
    else:
        I = torch.zeros(n_radius * n_angle**(d-1), *kernel_size, dtype=torch.float)
        kernel = torch.cartesian_prod(*[torch.arange(0, kernel_size[i], 1) for i in range(d)])
        for i in range(len(kernel)):
            f = torch.zeros(*kernel_size)
            f[tuple(kernel[i].tolist())] = 1
            I[(Ellipsis,) + tuple(kernel[i].tolist())] = torch.from_numpy(map_coordinates(f, sphere_coord, order=interpolation_order, mode='nearest'))

    return I.reshape(n_radius, -1, *I.shape[-d:])

###########################################################################################################################
############################################# Steerable Filter Basis ######################################################
###########################################################################################################################


def get_SHT_matrix(n_angle, freq_cutoff, dimension=2):
    '''
    Spherical Harmonic Transform Basis
    '''
    assert dimension in [2,3], "Only 2 and 3 dimensions are supported."
    if dimension == 2:
        SHT = (torch.fft.fft(torch.eye(freq_cutoff, n_angle)))
    
    if dimension == 3:
        theta, phi = torch.meshgrid(torch.pi * (torch.arange(n_angle)+0.5) / n_angle, 2 * torch.pi * torch.arange(n_angle) / n_angle, indexing='ij')
        factor = (2*torch.arange(n_angle//2) + 1)
        quadrature = torch.sin(theta) * (torch.sin(factor*theta.unsqueeze(-1)) / factor).sum(dim=-1)
        SHT = [torch.stack([torch.from_numpy(sph_harm(m, l, phi.numpy(), theta.numpy())).type(torch.cfloat)*sqrt(4*torch.pi/(2*l+1)) * quadrature
                            for m in range(-l, l+1)], dim=0).flatten(1)
               for l in range(freq_cutoff + 1)]
        
    return SHT 

def get_CG_matrix(dimension, freq_cutoff, n_angle=None):
    '''
    CG-Matrices
    '''
    assert dimension in [2,3], "Only 2 and 3 dimensions are supported."
    def get_CG_element(rho, rho1, rho2, freq_cutoff, n_angle=None, dimension=2):
        if dimension == 2:
            n_angle = n_angle if n_angle else freq_cutoff
            CG_tensor = torch.tensor([1 if (rho1+rho2-rho) % n_angle == 0 else 0])
        elif dimension == 3:
            n_angle = n_angle if n_angle else 2*(freq_cutoff + 1)
            CG_tensor = torch.zeros(2*rho+1,2*rho1+1, 2*rho2+1)
            if rho >= abs(rho1-rho2) and rho <= rho1+rho2:
                for m1 in range(-rho1, rho1+1):
                    for m2 in range(-rho2, rho2+1):
                        m = (m1+m2)
                        if abs(m) <= rho:
                            CG_tensor[m+rho,m1+rho1,m2+rho2] =  float(CG(rho1,m1,rho2,m2,rho,m).doit())
        return CG_tensor
    
    parts = freq_cutoff if dimension == 2 else freq_cutoff + 1
    C =[[[get_CG_element(rho, rho1, rho2, freq_cutoff, n_angle, dimension)
                  for rho2 in range(parts)]
              for rho1 in range(parts)]
         for rho in range(parts)]

    return C

def get_Fint_matrix(kernel_size, n_radius, n_angle, freq_cutoff, interpolation_type=1, sigma=0.6):
    '''
    Fusing Fourier (SHT) and Interpolation Matrix to give F-int matrix
    '''
    d = len(kernel_size)
    assert d in [2,3], "Only 2 and 3 dimensions are supported."
    assert -1 <= interpolation_type <= 5, "'interpolation_type' takes integer values between -1 and 5."
    if interpolation_type == -1:
        points = torch.stack(torch.meshgrid(*[torch.arange(-kernel_size[i]/2, kernel_size[i]/2, 1) + 0.5 for i in range(d)], indexing='xy'), dim=0)
        r = torch.linalg.vector_norm(points, dim=0)
        tau_r = torch.exp(-((r - torch.arange(1,n_radius+1).reshape(-1,*[1]*d))**2)/(2*(sigma**2))).type(torch.cfloat)
        tau_r[:,r==0] = 0
        if d == 2:
            tau_r[-1] = torch.exp(-((r - n_radius)**2)/(2*(0.4**2))).type(torch.cfloat)
            tau_r[-1,r==0] = 0
            theta = torch.arctan2(points[1], points[0])
            Fint = torch.stack([torch.exp( m * 1j * theta) for m in range(freq_cutoff)], dim=0)
            Fint = torch.einsum('rxy, mxy-> mrxy', tau_r, Fint)

        elif d == 3:
            theta = torch.nan_to_num(torch.acos(torch.clamp(points[2] / r, -1.0, 1.0)), nan=0.0)
            phi = torch.arctan2(points[1], points[0])
            Fint = []
            for l in range(freq_cutoff+1):
                Y_l = [torch.from_numpy(sph_harm(m, l, phi.numpy(), theta.numpy())).type(torch.cfloat) for m in range(-l, l + 1)]
                Fint.append(torch.stack(Y_l, dim=0).reshape(-1, 1, *kernel_size)*tau_r)
        
    elif 0 <= interpolation_type and interpolation_type<=5 and type(interpolation_type) == int:
        scalar = (torch.arange(1, n_radius+1)**(d-1)) / ((n_radius**d) * (n_angle**(d-1)))
        SHT = get_SHT_matrix(n_angle, freq_cutoff, d) # Spherical Harmonic Transform Matrix
        if d == 2:
            I = get_interpolation_matrix(kernel_size, n_radius, n_angle, interpolation_type).type(torch.cfloat) # Interpolation Matrix
            Fint = torch.einsum('r, mt, rtxy -> mrxy', scalar, SHT, I)
        elif d == 3:
            I = get_interpolation_matrix((kernel_size[2], kernel_size[0], kernel_size[1]), n_radius, n_angle, interpolation_type).type(torch.cfloat) # Interpolation Matrix
            I = torch.permute(I, (0,1,3,4,2))
            Fint = [torch.einsum('r, lt, rtxyz -> lrxyz', scalar, SHT[l], I) for l in range(freq_cutoff+1)] # Fint Matrix
    
    return Fint

def get_CFint_matrix(kernel_size, n_radius, n_angle, freq_cutoff_in, freq_cutoff_out, interpolation_type=1):
    '''
    Fusing Clebsch-Gordan Matrices with Fint Matrix (above) to give C-Fint matrix
    '''
    d = len(kernel_size)
    assert d in [2,3], "Only 2 and 3 dimensions are supported."
    assert -1 <= interpolation_type <= 5, "'interpolation_type' takes integer values between -1 and 5."
    freq_cutoff = max(freq_cutoff_in, freq_cutoff_out)
    Fint = get_Fint_matrix(kernel_size, n_radius, n_angle, freq_cutoff, interpolation_type)
    C = get_CG_matrix(d, freq_cutoff, n_angle)
    if d == 2:
        CFint = torch.einsum('lmn, nrxy -> lmrxy', torch.tensor(C, dtype=torch.cfloat), Fint)
        if interpolation_type>=0:
            CFint = CFint / freq_cutoff

    elif d == 3:
        if interpolation_type==-1:
            CFint = [[torch.stack([torch.einsum('lmn, nrxyz -> lrmxyz', C[l][l1][l2].type(torch.cfloat), Fint[l2])
                        for l2 in range(freq_cutoff+1)], dim=2)
                    for l in range(freq_cutoff_out+1)] 
                  for l1 in range(freq_cutoff_in+1)]

        if interpolation_type>=0:
            CFint = [[torch.stack([torch.einsum('lmn, nrxyz -> lrmxyz', C[l][l1][l2].type(torch.cfloat), Fint[l2]) / (freq_cutoff+1)
                        for l2 in range(freq_cutoff+1)], dim=2)
                    for l in range(freq_cutoff_out+1)]
                  for l1 in range(freq_cutoff_in+1)]

    return CFint

###########################################################################################################################
############################################# Steerable Positional Encoding ###############################################
###########################################################################################################################

def get_pos_encod(kernel_size, freq_cutoff):    
    d = len(kernel_size)
    points = torch.stack(torch.meshgrid(*[torch.arange(0, kernel_size[i], 1) for i in range(d)], indexing='ij'), dim=0).flatten(1)
    num_points = points.shape[-1]
    pairwise_diffs = points.unsqueeze(-1) - points.unsqueeze(1)
    pairwise_diffs = pairwise_diffs.view(d, -1)
    r_square = torch.sum(pairwise_diffs**2, dim=0)
    _ , indices = r_square.reshape(num_points, num_points).unique(return_inverse=True)

    if d == 2: 
        theta = torch.arctan2(pairwise_diffs[1], pairwise_diffs[0])
        pos_enc = torch.stack([torch.exp(-m *1j * theta) for m in range(freq_cutoff)], dim = 0)
        pos_enc = pos_enc.reshape(freq_cutoff, 1, num_points, 1, num_points)
        
    elif d == 3:
        theta = torch.nan_to_num(torch.arccos(torch.clamp(pairwise_diffs[2] / torch.sqrt(r_square), -1.0, 1.0)), nan=0.0)
        phi = torch.arctan2(pairwise_diffs[1], pairwise_diffs[0])
        pos_enc = []
        for l in range(freq_cutoff+1):
            part = torch.stack([sph_harm(m, l, phi, theta) for m in range(-l, l+1)], dim=-1)
            part = part.reshape(num_points, num_points, 2*l+1).transpose(-2,-1).unsqueeze(-2)
            pos_enc.append(part.type(torch.cfloat))
            
    return pos_enc, indices

#########################################################################################################################
####################################### Merge and Split Channels (3D) ###################################################
#########################################################################################################################

def merge_channel_dim(x, channel_last=False):
    if not channel_last:
        if type(x) is list:
            channels = [part.shape[2] for part in x]
            parts = [part.flatten(1,2) for part in x]
            x = torch.cat(parts, dim=1)
        else:
            channels = x.shape[-4]
    else:
        if type(x) is list:
            channels = [part.shape[-1] for part in x]
            parts = [part.flatten(-2,-1) for part in x]
            x = torch.cat(parts, dim=-1)
        else:
            channels = x.shape[-1]
    return x, channels

def split_channel_dim(x, channels, channel_last=False):
    split_index = 0 
    result = []
    if not channel_last:    
        for l,c in enumerate(channels):
            result.append(x[:, split_index:split_index + (2*l+1)*c].reshape(x.shape[0], 2*l+1, c, *x.shape[2:]))
            split_index += (2*l+1)*c
    else:
        for l,c in enumerate(channels):
            result.append(x[..., split_index:split_index + (2*l+1)*c].reshape(*x.shape[0], 2*l+1, c, *x.shape[2:]))
            split_index += (2*l+1)*c
    
    return result
