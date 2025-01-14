
import torch
import torch.nn as nn
from core.layers.SelfAttentionMemory import SelfAttentionMemory

class SpatioTemporalAttentionLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width,attention_hidden_dims, filter_size, stride, layer_norm):
        super(SpatioTemporalAttentionLSTMCell, self).__init__()

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
        self.attention_memory = SelfAttentionMemory(input_dim=num_hidden, hidden_dim=attention_hidden_dims)
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.W_cor =nn.Parameter(torch.zeros(num_hidden,  width, width))
        self.b_cor =nn.Parameter(torch.zeros(width))
    def map_interval(self,et, dim):
        T = 1.0 / torch.log(et+ 2.7183) #论文中把间隔除于了平均间隔,此处已经处理过
        ones = torch.ones(1, dim, dtype=torch.float32).to(et.device)
        T = T.matmul(ones).unsqueeze(2).unsqueeze(3)
        return T
    
    def forward(self, x_t, e_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

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

        c_new = f_t * c_t + i_t * g_t
        
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        #加入attention
        h_new,c_new,att_new=self.attention_memory(h_new,c_new)
        c_new=c_new*D_cor #经过自注意处理后的记忆进行折减考虑
        return h_new, c_new, m_new
