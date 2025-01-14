import torch
import torch.nn as nn
from core.attention.selfAttentionConvLSTM.BaseConvLSTMCell import ConvLSTMCell
from core.attention.selfAttentionConvLSTM.SelfAttention import SelfAttention
class AttPredRNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_heads,num_layers,kernel_size,padding):
        super(AttPredRNN,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.num_layers=num_layers
        self.kernel_size=kernel_size
        self.padding=padding
        self.conv2d=nn.Conv2d(input_dim,hidden_dim,kernel_size,padding=1,stride=0)
        self.selfAttention=nn.Sequential(
            SelfAttention(input_dim,num_heads),
            nn.LayerNorm(input_dim)
        ) 
    
    def forward(self,x,durations):
        feature_map=self.conv2d(x)

        x = self.norm_layer(x)
        x,weight=self.selfAttention(x,x,x)
