import torch
from math import sqrt
from Steerable.nn.Steerable3d.Steerable3d_pytorch.conv_layers import SE3BatchNorm, SE3NormNonLinearity
from Steerable.nn.utils import merge_channel_dim, get_pos_encod, split_channel_dim

#######################################################################################################################
############################################ SE(3) Multihead Self Attention ###########################################
#######################################################################################################################

class SE3MultiSelfAttention(torch.nn.Module):
    def __init__(self, transformer_dim, n_head, add_pos_enc=True):
        super(SE3MultiSelfAttention, self).__init__()

        self.query_dim = []
        self.n_head = n_head
        self.add_pos_enc = add_pos_enc
        self.transformer_dim = [transformer_dim] if type(transformer_dim) is not list and type(transformer_dim) is not tuple else transformer_dim
        self.maxl = len(self.transformer_dim) - 1
        self.scale = sqrt(sum([(2*l+1)*dim for l, dim in enumerate(self.transformer_dim)]))
        for dim in self.transformer_dim:
            if not dim % n_head == 0 :
                raise ValueError(f"Transformer dimension ({dim}) is not divisible by number of heads ({n_head}).")
            self.query_dim.append(dim // n_head)

        # Layer Parameters
        self.embeddings = torch.nn.ParameterList([torch.nn.Parameter(
                                        torch.randn(3, 1, 1, dim, dim, dtype = torch.cfloat))
                                        for dim in self.transformer_dim])
        self.out = torch.nn.ParameterList([torch.nn.Parameter(
                                        torch.randn(dim, dim, dtype = torch.cfloat))
                                        for dim in self.transformer_dim])
        self.pos_enc_weights = None
        self.pos_enc_basis = None
        self.radii_indices = None

    def intialize_positional_encoding(self, shape, maxl, device):
        with torch.no_grad():
            if self.pos_enc_basis is None and self.add_pos_enc:
                self.pos_enc_basis, self.radii_indices = get_pos_encod(shape, maxl)
                self.pos_enc_basis = [part.to(device) for part in self.pos_enc_basis]
            if self.pos_enc_weights is None:
                self.pos_enc_weights = torch.nn.ParameterList([torch.nn.Parameter(
                                        torch.randn(2, self.n_head, 1, dim, self.radii_indices.max(), 
                                                    dtype = torch.cfloat, device=device))
                                        for dim in self.query_dim])
                
            self.pos_weights = [torch.cat([
                    torch.zeros(2, self.n_head, 1, dim, 1, dtype = torch.cfloat, device=device),
                    self.pos_enc_weights[l]], dim=-1)
                    for l, dim in enumerate(self.query_dim)]
            
    def forward(self, x):
        x_shape = x[0].shape

        # Query Key Pair
        E, P = [], []
        if self.add_pos_enc:
            self.intialize_positional_encoding(x_shape[-3:], self.maxl, x[0].device)
        
        for l in range(self.maxl+1):
            # Embeddings
            E.append((self.embeddings[l] @ x[l].flatten(3)).reshape(3, x_shape[0], (2*l+1), self.n_head, self.query_dim[l], -1).transpose(2,3).flatten(3,4))
            # Attention Scores
            if self.add_pos_enc:
                P.append((self.pos_weights[l][..., self.radii_indices].movedim(-2, 2) * self.pos_enc_basis[l]).flatten(3,4))
        
        QKV = torch.cat(E, dim=3)
        Q, K, V = torch.conj(QKV[0].transpose(-2,-1)), QKV[1], QKV[2]
        A = (Q @ K) / self.scale
        if self.add_pos_enc:
            pos = torch.cat(P, dim=-2)
            A = A + (Q.unsqueeze(-2) @ pos[0]).squeeze(-2) / self.scale
        A = torch.nn.functional.softmax(A.abs(), dim=-1).type(torch.cfloat)
        
        V = V @ A.transpose(-2,-1)
        if self.add_pos_enc:
            V = V + (pos[1] @ A.unsqueeze(-1)).squeeze(-1).transpose(-2,-1)
        V = split_channel_dim(V.transpose(1,2), self.query_dim)
        
        # Output
        result = [(self.out[l] @ V[l].reshape(x_shape[0], 2*l+1, self.transformer_dim[l], -1)).reshape(*x[l].shape)
                  for l in range(self.maxl+1)]

        return result

#######################################################################################################################
################################################### SE(3) Transformer #################################################
#######################################################################################################################

class SE3PositionwiseFeedforward(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SE3PositionwiseFeedforward, self).__init__()

        self.input_dim = [input_dim] if type(input_dim) is not list and type(input_dim) is not tuple else input_dim
        self.hidden_dim = [hidden_dim] if type(hidden_dim) is not list and type(hidden_dim) is not tuple else hidden_dim
        
        assert len(self.hidden_dim) == len(self.input_dim)
        
        self.weights1 = torch.nn.ParameterList([torch.nn.Parameter(
                                        torch.randn(hidden, input, dtype = torch.cfloat))
                                         for input, hidden in zip(self.input_dim, self.hidden_dim)])
        self.nonlinearity = SE3NormNonLinearity(hidden_dim)

        self.weights2 = torch.nn.ParameterList([torch.nn.Parameter(
                                        torch.randn(input, hidden, dtype = torch.cfloat))
                                         for input, hidden in zip(self.input_dim, self.hidden_dim)])

    def forward(self, x):
        x_shape = x[0].shape
        x = [(self.weights1[l] @ part.flatten(3)).reshape(x_shape[0], 2*l+1, -1, *x_shape[3:]) for l,part in enumerate(x)]
        x = self.nonlinearity(x)
        x = [(self.weights2[l] @ part.flatten(3)).reshape(x_shape[0], 2*l+1, -1, *x_shape[3:]) for l,part in enumerate(x)]
         
        return x

#######################################################################################################################
################################################### SE(3) Transformer #################################################
#######################################################################################################################

class SE3Transformer(torch.nn.Module):
    def __init__(self, transformer_dim, n_head, hidden_dim, add_pos_enc=True):
        super(SE3Transformer, self).__init__()

        # Layer Design
        self.multihead_attention = SE3MultiSelfAttention(transformer_dim, n_head, add_pos_enc=add_pos_enc)
        self.positionwise_feedforward = SE3PositionwiseFeedforward(transformer_dim, hidden_dim)

        self.layer_norm1 = SE3BatchNorm()
        self.layer_norm2 = SE3BatchNorm()

    def forward(self, x):
        attentions = self.multihead_attention(self.layer_norm1(x))
        x = [attention + part for attention, part in zip(attentions, x)]
        postions = self.positionwise_feedforward(self.layer_norm2(x))
        x = [postion + part for postion, part in zip(postions, x)]

        return x

class SE3TransformerEncoder(torch.nn.Module):
    def __init__(self, transformer_dim, n_head, n_layers = 1, add_pos_enc=True):
        super(SE3TransformerEncoder, self).__init__()

        hidden_dim = [transformer_dim] if type(transformer_dim) is not list and type(transformer_dim) is not tuple else transformer_dim
        hidden_dim = [2*d for d in hidden_dim]
        # Layer Design
        self.transformer_encoder = torch.nn.Sequential(
            *[SE3Transformer(transformer_dim, n_head, hidden_dim, add_pos_enc=add_pos_enc) for _ in range(n_layers)]
        )
        self.norm = SE3BatchNorm()

    def forward(self, x):
        x = self.norm(self.transformer_encoder(x))
        
        return x

#######################################################################################################################
############################################## SE(3) Transformer Decoder ##############################################
####################################################################################################################### 

class SE3TransformerDecoder(torch.nn.Module):
    def __init__(self, transformer_dim, n_head, n_classes, n_layers, add_pos_enc=True):
        super(SE3TransformerDecoder, self).__init__()

        self.transformer_dim = transformer_dim
        self.n_classes = n_classes
        self.maxl = len(self.transformer_dim) - 1
        hidden_dim = [transformer_dim] if type(transformer_dim) is not list and type(transformer_dim) is not tuple else transformer_dim
        hidden_dim = [2*d for d in hidden_dim]
        self.transformer_encoder = torch.nn.Sequential(
            *[SE3Transformer(transformer_dim, n_head, hidden_dim, add_pos_enc=add_pos_enc) for _ in range(n_layers)]
        )

        self.class_embed = torch.nn.Parameter(torch.randn(1, 1, transformer_dim[0], n_classes, dtype=torch.cfloat))
        self.add_pos_enc = add_pos_enc
        self.pos_enc_basis = None
        self.radii_indices = None
        self.norm = SE3BatchNorm()
        
    def forward(self, x):
        if self.add_pos_enc and self.pos_enc_basis is None:
            pos_enc_basis, radii_indices = get_pos_encod(x[0].shape[-3:], self.maxl)
            pos_enc_basis = [part.to(x[0].device) for part in pos_enc_basis]
            self.pos_enc_basis = [torch.zeros(part.shape[0]+self.n_classes, part.shape[1], 1, part.shape[3]+self.n_classes, 
                                          dtype=torch.cfloat, device=x[0].device) for part in pos_enc_basis]
            self.radii_indices = torch.zeros(radii_indices.shape[0]+self.n_classes, radii_indices.shape[1]+self.n_classes, dtype=radii_indices.dtype, device=x[0].device)
            self.radii_indices[:radii_indices.shape[0],:radii_indices.shape[1]] = radii_indices.to(x[0].device)
            for l in range(len(pos_enc_basis)):
                self.pos_enc_basis[l][:pos_enc_basis[l].shape[0], :,:,:pos_enc_basis[l].shape[3]] = pos_enc_basis[l]
            for module in self.transformer_encoder:
                module.multihead_attention.pos_enc_basis = self.pos_enc_basis
                module.multihead_attention.radii_indices = self.radii_indices

        result = []
        x_shape = x[0].shape
        for l, part in enumerate(x):
            if l==0:
                class_embed = self.class_embed.expand(x[0].shape[0], -1, -1, -1)
            else:
                class_embed = torch.zeros(*part.shape[:3], self.n_classes, dtype=torch.cfloat, device=part.device)
            result.append(torch.cat([part.flatten(3), class_embed], dim=-1))
        
        result = self.transformer_encoder(result)
        patches, cls_seg_feat = [], []
        for l, part in enumerate(result):
            patches.append(part[..., : -self.n_classes].reshape(x_shape[0], 2*l+1, -1, *x_shape[3:]))
            cls_seg_feat.append(part[..., -self.n_classes :])
  
        return patches, cls_seg_feat

class SE3ClassEmbedings(torch.nn.Module):
    def __init__(self, transformer_dim, embedding_dim):
        super(SE3ClassEmbedings, self).__init__()
        
        self.transformer_dim = [transformer_dim] if type(transformer_dim) is not list and type(transformer_dim) is not tuple else transformer_dim
        self.embedding_dim = [embedding_dim] if type(embedding_dim) is not list and type(embedding_dim) is not tuple else embedding_dim
        
        assert len(self.transformer_dim) == len(self.embedding_dim)
        
        self.weight = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(dim1, dim2, dtype=torch.cfloat)) for dim1, dim2 in zip(self.embedding_dim, self.transformer_dim)
        ])
        self.norm = SE3BatchNorm()
        
    def forward(self, x, classes):
        classes = self.norm([self.weight[l] @ part for l, part in enumerate(classes)])
        x, _ = merge_channel_dim(x)
        classes, _ = merge_channel_dim(classes)
        result = (torch.conj(classes).transpose(-2,-1) @ x.flatten(2)).reshape(x.shape[0], classes.shape[-1], *x.shape[2:])
        
        return result
