import os
import torch
from torch.optim import Adam
from core.models import predrnn, predrnn_v2,predrnn_v3,predrnn_v32,MAU,predrnn_v4,predrnn_sa, predrnn_sa_v2,action_cond_predrnn, action_cond_predrnn_v2,convlstm_model

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.RNN,
            'predrnn_v2': predrnn_v2.RNN,
            'predrnn_v3': predrnn_v3.RNN,
            'predrnn_v32': predrnn_v32.RNN,
            'predrnn_v4': predrnn_v4.RNN,
            'predrnn_sa': predrnn_sa.RNN,
            'predrnn_sa_v2': predrnn_sa_v2.RNN,
            'action_cond_predrnn': action_cond_predrnn.RNN,
            'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
            'convlstm':convlstm_model.ConvLSTM_Model,
            'mau':MAU.RNN
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        save_path=f"{self.configs.save_dir}/{self.configs.model_name}_{self.configs.dataset_name}_{self.configs.data_type}_{self.configs.rand_or_fix}_{self.configs.img_width}_{self.configs.px}_{self.configs.total_length}"
        checkpoint_path = os.path.join(save_path, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path, map_location=torch.device(self.configs.device))#加载到指定的device
        self.network.load_state_dict(stats['net_param'])

    def trainV3(self,frames,distance,mask):
        frames_tensor = frames.to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        distance_tensor= distance.to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames, loss = self.network(frames_tensor,distance_tensor, mask_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()
    
    def train(self, frames, mask):
        frames_tensor = frames.to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames, loss = self.network(frames_tensor, mask_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def testV3(self, frames,distance, mask):
        frames_tensor = frames.to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        distance_tensor= distance.to(self.configs.device)
        next_frames, _ = self.network(frames_tensor,distance_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()
    
    def test(self, frames, mask):
        frames_tensor = frames.to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()