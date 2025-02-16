'''
Author: Ahmed Zgaren
Date: Feb 2024
'''
import torch
from torch import Tensor, nn
from .regressor import RegNet
import math
from torch.autograd import Variable
from .cotLayer import CotLayer
from transformers import DeiTForImageClassification


class Trans(nn.Module):
    def __init__(self, d_model,h, d_ff, num_layers, enc_in, drop = 0.1):
        super(Trans, self).__init__()
        #VEM
        self.encoder = nn.Sequential(
            CotLayer(enc_in, 3),
            nn.Conv2d(enc_in, enc_in,3,1),
            nn.MaxPool2d(2),
            CotLayer(enc_in,3),
            nn.Conv2d(enc_in, enc_in,3,1),
            nn.MaxPool2d(2),
            CotLayer(enc_in,3),
            nn.Conv2d(enc_in, enc_in,3,1),
            CotLayer(enc_in,3),
            nn.Conv2d(enc_in, d_model,3,1)
   
        )
        #SAMM
        self.transformer = Transformer(d_model)
        #CRM
        self.count = RegNet(d_model, dropout = drop)
        self.flatten = nn.Flatten(1,-1)
        self.gap = nn.AvgPool1d(196)
        
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        self.apply(_weights_init)
            
    def forward(self, x):
        #x: feature maps from backbone (shape: Batch,enc_in, width, height; batch, 256,80,80)
        z = self.encoder(x) # shape: batch x d_model=768 x 14 x 14
        out  = z.reshape(z.shape[0], z.shape[1],z.shape[2]*z.shape[3]) # shape: batch x 768 x 196
        out = out.permute(0,2,1) #shape: batch x 196 x 768 
        out, tmap = self.transformer(out) # tmap shape: batch x 197 x 768
        out = self.gap(tmap[0][:,1:].permute(0,2,1)) # shape: batch x 1 x 197
        out = self.flatten(out) #shape: batch x 197
        
        out = self.count(out)
        
        return out, tmap, z
    
class Transformer(nn.Module):
    '''
    SAMM class definition 
    '''
    def __init__(self, d_model):
        super(Transformer, self).__init__()
        #import pretrained ViT
        DEIT = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224')
        #create the learnable token
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.transformer_encoder = DEIT.deit.encoder
            
    def forward(self, x):
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        out = self.transformer_encoder(x, output_attentions = True)

        return out[0][:, 0], out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)] # type: ignore
        return self.dropout(x)
