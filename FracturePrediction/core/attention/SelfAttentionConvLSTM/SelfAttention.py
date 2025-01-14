import torch
import torch.nn as nn
import numpy as np
from core.utils.constants import DEVICE
class SelfAttention(nn.Module):
    """Self-Attention module implementation."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(SelfAttention, self).__init__()
        self.query_h = nn.Conv2d(input_dim, hidden_dim, 1, padding="same")
        self.key_h = nn.Conv2d(input_dim, hidden_dim, 1, padding="same")
        self.value_h = nn.Conv2d(input_dim, input_dim, 1, padding="same")
        self.z = nn.Conv2d(input_dim, input_dim, 1, padding="same")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, h):
        batch_size, _, H, W = h.shape
        k_h = self.key_h(h)
        q_h = self.query_h(h)
        v_h = self.value_h(h)

        k_h = k_h.view(batch_size, self.hidden_dim, H * W)
        q_h = q_h.view(batch_size, self.hidden_dim, H * W).transpose(1, 2)
        v_h = v_h.view(batch_size, self.input_dim, H * W)
        torch.cuda.empty_cache()
        # attention_temp = torch.softmax(
        #     torch.bmm(q_h.to("cuda:2"), k_h.to("cuda:2")), dim=-1
        # ).to("cuda:2")#放到其他单独的显卡上算
        attention = torch.softmax(
            torch.bmm(q_h, k_h), dim=-1
        ) # the shape is (batch_size, H*W, H*W) 显卡扛不住，需要的内参太高了
        new_h = torch.matmul(attention, v_h.permute(0, 2, 1))
        new_h = new_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        new_h = self.z(new_h)

        return new_h, attention