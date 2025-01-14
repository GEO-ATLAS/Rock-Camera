from core.attention.SelfAttentionConvLSTM.SASeq2Seq import SASeq2Seq
from core.attention.SelfAttentionMemoryConvLSTM.SAMSeq2Seq import SAMSeq2Seq
import torch
import os
from torch.optim import Adam
import torch.nn as nn

class Model(object):
    def __init__(self,configs) -> None:
        self.configs = configs
        networks_map = {
            'SASeq2Seq': SASeq2Seq,
            'SAMSeq2Seq': SAMSeq2Seq,
        }
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(attention_hidden_dims=configs.attention_hidden_dims,
            num_channels=configs.img_channel,
            kernel_size=configs.filter_size,
            num_kernels=configs.num_kernels,
            padding="same",
            activation=configs.activation,
            frame_size=(configs.img_width,configs.img_width),
            num_layers=configs.num_layers,
            input_seq_length=configs.input_length,
            return_sequences=True).to(configs.device)
            #self.network=nn.DataParallel(self.network,device_ids=[0,1,2,3])#多卡并行,第一个时Contants中的Device
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.MSE_criterion = nn.MSELoss()

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        save_path=self.configs.save_dir+"/"+self.configs.model_name+"_"+self.configs.dataset_name
        os.makedirs(save_path,exist_ok=True)
        checkpoint_path = os.path.join(save_path, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)
    
    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path, map_location=torch.device(self.config.device))#加载到指定的device
        self.network.load_state_dict(stats['net_param'])
    
    def train(self, frames):
        self.optimizer.zero_grad()
        input_frames=frames[:,:,:self.configs.input_length]
        y_real=frames[:,:,self.configs.input_length:]
        next_frames= self.network(input_frames)
        loss=self.MSE_criterion(next_frames,y_real)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()
    
    def test(self, frames):
        input_frames=frames[:,:,:self.configs.input_length]
        torch.cuda.empty_cache()
        next_frames= self.network(input_frames)
        return next_frames