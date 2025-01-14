import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
dataRoot="./DataSet/"#set your DataSet
class DataTypeEnum:
    L255 = "255"
    L204 = "204"
    L51x5 = "51x5"
    L51x4 = "51x4"
    LCaptureImg="captureImage"
    LSliceImg="sliceImage"
    LCaptureColorImg="captureColorImage"
    LSliceColorImg="sliceColorImage"
    #3d line
    L3d306 = "3d306"
    L3d51x4 = "3d51x4"
    L3d51x6 = "3d51x6"
    L3d204 = "3d204"
    Lcal="center_angle_length"#sort by angle
    Lcal_map="cal_map"
    Lcal_gauss_map="cal_gauss_map"
    Lcap2slice="capture2slice"
    Lcap2sliceColor="cap2sliceColor"
    LcapsliceFusion="capsliceFusion"


# 创建一个虚构的数据集类
class CustomDataset(Dataset):
    def __init__(self, dataSetName,dataRoot=dataRoot,dataType=DataTypeEnum.L255,
            randOrFix=True,useClips=True,withChannel=False,
            ChannelLast=False,
            px=2,#裂隙的像素宽度
            size=128,#图像尺寸
            sequence_length=16,#这个是总长度根据需要更改
            input_length=8,
            sequence_offset=0,randFlag=1):
        '''
            useClips:是否使用切片
        '''
        if randOrFix:
            #随机模式1：正态分布，均值1，方差1，取绝对值
            if randFlag==1:
                dataPath=dataRoot+"/NRandStep/"+dataSetName+'/'
            #随机模式2:均值1.5，正负0.25，均匀分布
            if randFlag==2:
                dataPath=dataRoot+"/NRandStep2/"+dataSetName+'/'
        else:
            dataPath=dataRoot+"/FixStep05/"+dataSetName+'/'
        self.useClips=useClips
        self.clips=np.load(dataPath+f"frame_clips_{sequence_length}-train.npy") #切片索引
        
        
        if dataType not in DataTypeEnum.__dict__.values():
            raise ValueError("数据类型必须从"+DataTypeEnum.__dict__.values()+"选择")
        
        if dataType==DataTypeEnum.LCaptureImg:
            self.frames=np.load(dataPath+f"fracture_images_raw_{size}_{px}px.npy") #加载2个像素宽度的图片

        if dataType==DataTypeEnum.LSliceImg:
            self.frames=np.load(dataPath+f"fracture_slice_images_raw_{size}_{px}px.npy")

        if dataType==DataTypeEnum.Lcap2slice:
            self.frames_slice=np.load(dataPath+f"fracture_slice_images_raw_{size}_{px}px.npy")
            self.frames_cap=np.load(dataPath+f"fracture_images_raw_{size}_{px}px.npy") #加载2个像素宽度的图片

        if dataType==DataTypeEnum.Lcap2sliceColor:
            self.frames_slice=np.load(dataPath+f"fracture_images_raw_{size}_{px}px_color.npy")
            self.frames_cap=np.load(dataPath+f"fracture_slice_images_raw_{size}_{px}px_color.npy")

        self.positions=np.load(dataPath+"frame_position-train.npy")
        #固定一个clip8帧，可以抽取其中的帧数，满足从0开始，sequence_offset+sequence_length<8即可
        #一个clip至少2帧,offset:0,1,2,3,4,5,6,length:8,7,6,5,4,3,2,共7中组合
        self.input_length=input_length
        self.sequence_offset=sequence_offset 
        self.sequence_length=sequence_length
        self.with_channel=withChannel #
        self.dataType=dataType
        self.channel_last=ChannelLast

    def __len__(self):
        if self.useClips:
            return len(self.clips)
        else:
            return len(self.frames)

    def __getitem__(self, idx):
        durations=[]
        if self.useClips:
            begin = self.clips[idx, 0]
            end =  self.clips[idx, 1]
            frames_slice=self.frames_slice[begin:end,:]
            frames_cap=self.frames_cap[begin:end,:]
            position_slice=self.positions[begin:end]
            first_value=position_slice[0]
            durations_to_first=position_slice-first_value
            durations_to_first = durations_to_first[:, np.newaxis]
            durations=torch.tensor(durations_to_first, dtype=torch.float32)

        if self.dataType==DataTypeEnum.Lcap2slice:
            #frames_slice=frames_slice[:,:,np.newaxis] #添加通道维度
            frames_slice=frames_slice[:,:,np.newaxis]
            frames_cap=frames_cap[:,:,np.newaxis]
            frames_slice=np.transpose(frames_slice, (0,1,3,2))
            frames_cap=np.transpose(frames_cap, (0,1,3,2))

        #sequences=np.concatenate([frames_cap,frames_slice],axis=0)
        #print(sequences.shape)     

        sequences_cap=torch.tensor(frames_cap, dtype=torch.float32)
        sequences_slice=torch.tensor(frames_slice, dtype=torch.float32)
        # if self.dataType==DataTypeEnum.LCaptureColorImg or self.dataType==DataTypeEnum.LSliceColorImg:
        #     sequences=sequences.permute(0,3,1,2)#c,s,h,w
        if self.input_length==0:
            sequences_con=sequences_slice # all slice
            
        if self.input_length>0 and self.input_length<self.sequence_length:

            sequences_con=np.concatenate([sequences_cap[:,:self.input_length],sequences_slice[:,self.input_length:]],axis=1)

        if self.input_length==self.sequence_length:
            sequences_con=sequences_cap # all cap

        sample = {
            'sequences': sequences_con,
            'durations': durations
        }
        return sample

class ImagePairsWithDistance(Dataset):
    def __init__(self,dataSetName="PBSet1",
                 randOrFix=True,
                 useColor=False,
                 useSliced=False,
                 px=2,size=128
                 ) -> None:
        super().__init__()
        if randOrFix:
            dataPath=dataRoot+"/NRandStep/"+dataSetName+'/'
        else:
            dataPath=dataRoot+"/FixStep05/"+dataSetName+'/'
        self.pairs=np.load(dataPath+"fracture_images_pairs_width_distance.npy")
        if useSliced:
            if useColor:
                self.image_raw=np.load(dataPath+f"fracture_slice_images_raw_{size}_{px}px_color.npy")
            else:
                self.image_raw=np.load(dataPath+f"fracture_slice_images_raw_{size}_{px}px.npy")
        else:
            if useColor:
                self.image_raw=np.load(dataPath+f"fracture_images_raw_{size}_{px}px_color.npy")
            else:
                self.image_raw=np.load(dataPath+f"fracture_images_raw_{size}_{px}px.npy")
                
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        begin = self.pairs[idx, 0]
        end =  self.pairs[idx, 1]
        distance=self.pairs[idx, 2]
        begin = int(begin)
        end =   int(end)
        sample = {
            'input': torch.tensor(self.image_raw[begin,:], dtype=torch.float32),
            'goal': torch.tensor(self.image_raw[end,:], dtype=torch.float32),
            'distance':torch.tensor(np.array([distance]), dtype=torch.float32)
        }
        return sample

class ImageDataSet(Dataset):
    def __init__(self,dataSetName="PBSet1",randOrFix=True,useSliced=False,px=2,size=128) -> None:
        super().__init__()
        if randOrFix:
            dataPath=dataRoot+"/NRandStep/"+dataSetName+'/'
        else:
            dataPath=dataRoot+"/FixStep05/"+dataSetName+'/'
        if useSliced:
            self.image_raw=np.load(dataPath+f"fracture_slice_images_raw_{size}_{px}px.npy")
        else:
            self.image_raw=np.load(dataPath+f"fracture_images_raw_{size}_{px}px.npy")
    def __len__(self):
        return len(self.image_raw)

    def __getitem__(self, idx):
        sample = {
            'image': torch.tensor(self.image_raw[idx,:], dtype=torch.float32),
        }
        return sample
class CustomDataLoader:
   def __init__(self,dataSet:CustomDataset,train_batch_size=32,test_batch_size=32,train_ratio=0.9,shuffle=True):
        #split_idx = int(len(dataSet) * train_ratio)
        #train_indices = list(range(split_idx))
        #test_indices = list(range(split_idx, len(dataSet)))
        total=len(dataSet)
        index=list(range(total))
        train_num=int(total*train_ratio)
        train_indices=random.sample(index,train_num)
        test_indices=list(set(index)-set(train_indices))#[x for x in index if x not in train_indices]
        # 使用Subset创建训练集和测试集的子集,随机采样
        train_dataset = Subset(dataSet, train_indices)
        test_dataset = Subset(dataSet, test_indices)
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,shuffle=shuffle)
        self.test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,shuffle=shuffle)
        print("data prepare finished.....")

if __name__=="__main__":
    # 示例用法
    dataset = CustomDataset(dataSetName="PBSet1", dataType=DataTypeEnum.Lcap2slice,randOrFix=True,useClips=True,sequence_length=16,input_length=8,px=1,size=128,ChannelLast=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # for batch in dataloader:
    #     sequences = batch['sequences']
    #     durations = batch['durations']
    #     print(sequences.shape,durations.shape)
    # 计算划分的数据集大小
    num_samples = len(dataset)
    # 划分训练集和测试集
    split_ratio = 0.9  # 训练集占总数据的80%
    split_idx = int(len(dataset) * split_ratio)
    train_indices = list(range(split_idx))
    test_indices = list(range(split_idx, len(dataset)))

    # 使用Subset创建训练集和测试集的子集
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # 使用采样器创建训练集和测试集的 DataLoader
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    b=next(iter(test_dataset))
    print(b)
    print(num_samples,len(train_dataset),len(test_dataset))


