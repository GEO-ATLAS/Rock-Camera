#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2023/09/16 09:29:37
@Author      :chenbaolin
@version      :1.0
'''
import torch
import torch.nn as nn

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.W_cor =nn.Parameter(torch.zeros(num_hidden,  width, width))
        self.b_cor =nn.Parameter(torch.zeros(width))
        nn.init.uniform_(self.W_cor)
        nn.init.normal_(self.b_cor)
        self.sigmoid=nn.Sigmoid()

    def map_interval(self,et, dim):
        #除于均值的必要性存疑，鸡肋，相对性？
        #这里平均的取值有误，比较复杂，应当是同一序列里的平均间隔，暂时先这么处理
        T = 1.0 / torch.log(et+2.7183) #论文中把间隔除于了平均间隔，此处在输入前已经处理
        ones = torch.ones(1, dim, dtype=torch.float32).to(et.device)
        T = T.matmul(ones).unsqueeze(2).unsqueeze(3)
        return T

    def forward(self, x_t, e_t , h_t, c_t, m_t):
        '''增加间隔信息'''
        
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        #
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        #注意与隐层数量保持一致
        E_t=self.map_interval(e_t,self.num_hidden)
        #这里Time-Aware论文里用的是tanh,但他们开源的代码里用的sigmoid
        D_ST=torch.tanh(self.W_cor*c_t+self.b_cor)##c_t就是xue论文中对应的D_t-1上一个状态
        D_ST_dis=D_ST*E_t #对短时记忆打折 #xue论文里合并在一起了
        #long term memory
        D_long=c_t-D_ST
        D_cor=D_ST_dis+D_long #D_cor=E_t*D_ST+c_t-D_ST
        #print("D_cor",E_t,D_cor.shape,c_t.shape,f_t.shape,torch.max(D_cor),torch.min(D_cor))
        c_new = f_t * D_cor + i_t * g_t #Dt

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new
    
if __name__=="__main__":
    #an test
    def test_spatiotemporal_lstm_cell():
        # Define input dimensions
        batch_size = 16
        in_channels = 3
        width = 128 #patch_size width
        num_hidden = 64

        # Create an instance of the SpatioTemporalLSTMCell
        lstm_cell = SpatioTemporalLSTMCell(in_channels, num_hidden, width, filter_size=3, stride=1, layer_norm=False)
        lstm_cell2 = SpatioTemporalLSTMCell(num_hidden, num_hidden, width, filter_size=3, stride=1, layer_norm=False)
        # Generate sample input data
        x_t = torch.randn(batch_size, in_channels, width, width)
        e_t = torch.ones(batch_size, 1)
        h_t = torch.randn(batch_size, num_hidden, width, width)
        c_t = torch.randn(batch_size, num_hidden, width, width)
        m_t = torch.randn(batch_size, num_hidden, width, width)

        # Forward pass through the LSTM cell
        h_new, c_new, m_new = lstm_cell(x_t, e_t, h_t, c_t, m_t)
        
        print(h_new.shape,c_new.shape,m_new.shape)
        h_new, c_new, m_new = lstm_cell2(h_new, e_t, h_new,c_new, m_new)
        # Validate the output dimensions
        assert h_new.shape == (batch_size, num_hidden, width, width)
        assert c_new.shape == (batch_size, num_hidden, width, width)
        assert m_new.shape == (batch_size, num_hidden, width, width)

        print("SpatioTemporalLSTMCell test passed.")
    test_spatiotemporal_lstm_cell()