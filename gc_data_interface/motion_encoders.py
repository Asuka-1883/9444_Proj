"""Three interchangeable motion encoders: [B,T,4] -> (sequence, [L,B,H])."""
import math
import torch
from torch import nn
from lstm_encoder import LSTMMotionEncoder




class TransformerMotionEncoder(nn.Module):
    def __init__(self,hidden_size=64,layers=2,heads=4,ffn=128,dropout=.1,max_steps=128):
        super().__init__()
        if hidden_size%heads:
            raise ValueError('Transformer hidden_size must be divisible by heads')
        self.projection=nn.Linear(4,hidden_size)
        positions=torch.arange(max_steps,dtype=torch.float32)[:,None]
        frequency=torch.exp(torch.arange(0,hidden_size,2,dtype=torch.float32)*(-math.log(10000.)/hidden_size))
        encoding=torch.zeros(max_steps,hidden_size)
        encoding[:,0::2]=torch.sin(positions*frequency)
        encoding[:,1::2]=torch.cos(positions*frequency[:hidden_size//2])
        self.register_buffer('positions',encoding[None])
        layer=nn.TransformerEncoderLayer(d_model=hidden_size,nhead=heads,dim_feedforward=ffn,
                    dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=layers,
                     norm=nn.LayerNorm(hidden_size),enable_nested_tensor=False)
        # Encoder layers are independent and receive independent initial weights.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim>1:nn.init.xavier_uniform_(parameter)

    def forward(self,x):
        if x.shape[1]>self.positions.shape[1]:
            raise ValueError('Observed sequence exceeds positional encoding capacity')
        sequence=self.encoder(self.projection(x)+self.positions[:,:x.shape[1]])
        # All tokens are observed history. Bidirectional attention introduces no future leakage.
        return sequence,sequence[:,-1].unsqueeze(0)


def make_motion_encoder(kind='gru',hidden_size=64,layers=None,heads=4,ffn=128,dropout=.1):
    kind=kind.lower()
    if kind not in ('gru','lstm','transformer'):
        raise ValueError('encoder_type must be gru, lstm or transformer')
    layers=layers if layers is not None else (2 if kind=='transformer' else 1)
    if hidden_size<1 or layers<1 or heads<1 or ffn<1 or not 0<=dropout<1:
        raise ValueError('Invalid encoder dimensions or dropout')
    if kind=='gru':
        # Retain native GRU state_dict names for existing trained weights.
        return nn.GRU(4,hidden_size,num_layers=layers,batch_first=True,
                      dropout=dropout if layers>1 else 0.)
    if kind=='lstm':return LSTMMotionEncoder(hidden_size,layers,dropout)
    return TransformerMotionEncoder(hidden_size,layers,heads,ffn,dropout)
