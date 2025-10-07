import torch 
from math import sqrt

from Steerable.nn.utils import get_pos_encod
from Steerable.nn.Steerable2d.conv_layers import SE2NormNonLinearity, SE2BatchNorm
        
#######################################################################################################################
############################################## Multihead Self Attention ###############################################
#######################################################################################################################

class SE2MultiSelfAttention(torch.nn.Module):
    def __init__(self, transformer_dim, n_head, freq_cutoff, add_pos_enc = True):
        super(SE2MultiSelfAttention, self).__init__()

        # Layer Design
        if not transformer_dim % n_head == 0 :
            raise ValueError(f"Transformer dimension ({transformer_dim}) is not divisible by number of heads ({n_head}).")
        self.query_dim = transformer_dim // n_head
        self.n_head = n_head
        self.freq_cutoff = freq_cutoff
        self.add_pos_enc = add_pos_enc

        # Layer Parameters
        self.embeddings = torch.nn.Parameter(torch.randn(3, 1, freq_cutoff, n_head, self.query_dim, transformer_dim, dtype = torch.cfloat))
        self.out = torch.nn.Parameter(torch.randn(freq_cutoff, transformer_dim, n_head * self.query_dim, dtype = torch.cfloat))

        self.pos_enc_weights = None
        self.pos_enc_basis = None
        self.radii_indices = None
        
    def initialize_parameters(self, shape, freq_cutoff, device):
        with torch.no_grad():
            if self.pos_enc_basis is  None or self.radii_indices is None:
                    self.pos_enc_basis, self.radii_indices = get_pos_encod(shape, freq_cutoff)
                    self.pos_enc_basis = self.pos_enc_basis.to(device)
            if self.pos_enc_weights is None:
                    self.pos_enc_weights = torch.nn.Parameter(torch.randn(2, self.freq_cutoff, self.n_head, self.query_dim, self.radii_indices.max(), dtype=torch.cfloat, device=device))
                    
        self.pos_weights = torch.cat([torch.zeros(2, self.freq_cutoff, self.n_head, self.query_dim, 1, dtype = torch.cfloat, device=device),
                    self.pos_enc_weights],dim=-1)
            
                

    def forward(self, x):
        x_shape = x.shape
        x = x.flatten(3) # shape : batch x freq_cutoff x channel x N
        
        # Query, Key and Value Embeddings
        E = (self.embeddings @ x.unsqueeze(2))
        Q, K, V = torch.conj(E[0].transpose(-2,-1)), E[1], E[2]
        
        # Scores
        A = Q @ K
        if self.add_pos_enc:
            self.initialize_parameters(x_shape[-2:],self.freq_cutoff, x.device)
            pos = self.pos_weights[..., self.radii_indices].transpose(-3,-2) * self.pos_enc_basis
            A = A + (Q.unsqueeze(-2) @ pos[0]).squeeze(-2)
        
        # Attention Weights
        A = torch.sum(A, dim=1, keepdim=True)
        A = torch.nn.functional.softmax(A.abs() / sqrt(self.query_dim), dim = -1).type(torch.cfloat)
 
        # Output
        result = V @ A.transpose(-2,-1)
        if self.add_pos_enc:
            result = result + (pos[1] @ A.unsqueeze(-1)).squeeze(-1).transpose(-2,-1)
        
        # Mixing Heads
        result = self.out @ result.flatten(2,3)
        result = result.reshape(*x_shape)

        return result

#######################################################################################################################
############################################## Positionwise Feedforward ###############################################
#######################################################################################################################    

class SE2PositionwiseFeedforward(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, freq_cutoff):
        super(SE2PositionwiseFeedforward, self).__init__()

        self.freq_cutoff = freq_cutoff

        self.weights1 = torch.nn.Parameter(torch.randn(freq_cutoff, hidden_dim, input_dim, dtype = torch.cfloat))
        self.weights2 = torch.nn.Parameter(torch.randn(freq_cutoff, input_dim, hidden_dim, dtype = torch.cfloat))
        
        self.eps = 1e-5
        self.nonlinearity = SE2NormNonLinearity(hidden_dim, freq_cutoff, nonlinearity=torch.nn.GELU())

    def forward(self, x):
        x_shape = x.shape
        x = x.flatten(3) # shape : batch x freq_cutoff x channel x N

        x = self.weights1 @ x
        x = self.nonlinearity(x)
        x = self.weights2 @ x
        
        x = x.reshape(*x_shape)
        return x   

#######################################################################################################################
################################################# SE(2) Transformer ###################################################
####################################################################################################################### 

class SE2Transformer(torch.nn.Module):
    def __init__(self, transformer_dim, n_head, hidden_dim, freq_cutoff, add_pos_enc = True):
        super(SE2Transformer, self).__init__()

        # Layer Design
        self.multihead_attention = SE2MultiSelfAttention(transformer_dim, n_head, freq_cutoff, add_pos_enc)
        self.positionwise_feedforward = SE2PositionwiseFeedforward(transformer_dim, hidden_dim, freq_cutoff)

        self.layer_norm1 = SE2BatchNorm()
        self.layer_norm2 = SE2BatchNorm()

    def forward(self, x):
        x = self.multihead_attention(self.layer_norm1(x)) + x
        x = self.positionwise_feedforward(self.layer_norm2(x)) + x
 
        return x
    
#######################################################################################################################
############################################## SE(2) Transformer Encoder ##############################################
####################################################################################################################### 

class SE2TransformerEncoder(torch.nn.Module):
    def __init__(self, transformer_dim, n_head, freq_cutoff, n_layers = 1, add_pos_enc = True):
        super(SE2TransformerEncoder, self).__init__()

        # Layer Design
        self.transformer_encoder = torch.nn.Sequential(
            *[SE2Transformer(transformer_dim, n_head, 2*transformer_dim, freq_cutoff, add_pos_enc) for _ in range(n_layers)]
        )
        self.norm = SE2BatchNorm()

    def forward(self, x):
        x = self.norm(self.transformer_encoder(x))
        return x
    
#######################################################################################################################
############################################## SE(2) Transformer Decoder ##############################################
####################################################################################################################### 

class SE2TransformerDecoder(torch.nn.Module):
    def __init__(self, transformer_dim, n_head, freq_cutoff, n_classes, n_layers, add_pos_enc = True):
        super(SE2TransformerDecoder, self).__init__()

        self.transformer_dim = transformer_dim
        self.scale = transformer_dim ** -0.5
        self.n_head = n_head
        self.n_classes = n_classes
        self.freq_cutoff = freq_cutoff
        
        self.transformer_encoder = torch.nn.Sequential(
            *[SE2Transformer(transformer_dim, n_head, 2*transformer_dim, freq_cutoff, add_pos_enc) for _ in range(n_layers)]
        )

        self.class_embed = torch.nn.Parameter(torch.randn(1, 1, transformer_dim, n_classes, dtype=torch.cfloat))
        self.C = torch.tensor([[[(m1+m2-m)%freq_cutoff == 0 for m2 in range(freq_cutoff)]
                           for m1 in range(freq_cutoff)] for m in range(freq_cutoff)]).type(torch.cfloat)
        self.add_pos_enc = add_pos_enc
        self.pos_enc_basis = None
        self.radii_indices = None
    def forward(self, x):
        # Positional Encoding
        if self.add_pos_enc and (self.pos_enc_basis == None or self.radii_indices == None):
            pos_enc_basis, radii_indices = get_pos_encod(x.shape[-2:], self.freq_cutoff)
            self.pos_enc_basis = torch.zeros(self.freq_cutoff, 1, pos_enc_basis.shape[2]+self.n_classes, 1, pos_enc_basis.shape[4]+self.n_classes, dtype=torch.cfloat, device=x.device)
            self.pos_enc_basis[:,:,:pos_enc_basis.shape[2],:,:pos_enc_basis.shape[4]] = pos_enc_basis.to(x.device)
            self.radii_indices = torch.zeros(radii_indices.shape[0]+self.n_classes, radii_indices.shape[1]+self.n_classes, dtype=radii_indices.dtype, device=x.device)
            self.radii_indices[:radii_indices.shape[0],:radii_indices.shape[1]] = radii_indices.to(x.device)
            for module in self.transformer_encoder:
                module.multihead_attention.pos_enc_basis = self.pos_enc_basis
                module.multihead_attention.radii_indices = self.radii_indices
        
        # Transformer
        x_shape = x.shape
        pad = torch.zeros(x_shape[0], self.freq_cutoff-1, self.transformer_dim, self.n_classes, dtype=torch.cfloat, device=self.class_embed.device)
        class_embed = torch.cat((self.class_embed.expand(x_shape[0], 1, -1, -1), pad), dim=1)
        x = torch.cat((x.flatten(3), class_embed), -1)
        x = self.transformer_encoder(x)

        # Masks        
        patches, cls_seg_feat = x[..., : -self.n_classes], x[..., -self.n_classes :]
        patches = patches.reshape(x_shape[0], self.freq_cutoff, -1, *x_shape[3:])
        
        return patches, cls_seg_feat

class SE2ClassEmbeddings(torch.nn.Module):
    def __init__(self, transformer_dim, embedding_dim, freq_cutoff):
        super(SE2ClassEmbeddings, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(freq_cutoff, embedding_dim, transformer_dim, dtype=torch.cfloat))
        self.norm = SE2BatchNorm()

    def forward(self, x, classes):
        classes = self.norm(self.weight @ classes).flatten(1,2)
        result = torch.conj(classes).transpose(-2,-1) @ x.flatten(1,2).flatten(2)
        result = result.reshape(x.shape[0], classes.shape[-1], *x.shape[-2:])
        
        return result