import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
def LineCount(sequences):
    length_slice=[]
    for i in range(sequences.shape[2] // 5):
        start_idx = i * 5
        slice3 = sequences[:,:, start_idx+4].unsqueeze(2)
        length_slice.append(slice3)
    #[batch,seq,feature]
    length_flat = torch.cat(length_slice, dim=2)
    #[batch,seq_sum]
    length_greater_than_zero = (length_flat > 0).sum(dim=2).float()
    return length_greater_than_zero

def LineCountByPoints(sequences):
    '''计算线段长度以及线段大于零的线段，对是否为线段进行了排序'''
    batch_size, sequence_length, num_features = sequences.shape
    num_segments = num_features // 4
    line_lengths = torch.zeros((batch_size, sequence_length, num_segments))
    for i in range(num_segments):
        p1 = sequences[:, :, 4 * i : 4 * i + 2]
        p2 = sequences[:, :, 4 * i + 2 : 4 * i + 4]
        
        # 计算线段长度
        lengths = torch.norm(p2 - p1, dim=2, keepdim=True)  # 使用PyTorch计算欧几里得范数来得到线段长度，并保持维度
        line_lengths[:, :, i] = lengths.squeeze(-1)
    line_lengths=line_lengths.permute(2, 0, 1)
    valid_lines = (line_lengths > 0).int()
    #sorted_lines, _ = torch.sort(valid_lines, dim=2, descending=True)
    total_length=line_lengths.sum(dim=2).unsqueeze(-1)
    #计算各个
    return valid_lines,line_lengths,total_length
        
def count(sequences):
    centers_slice = []
    directions_slice = []
    length_slice=[]
    for i in range(sequences.shape[2] // 5):
        start_idx = i * 5
        slice1 = sequences[:,:, start_idx:start_idx+2]
        centers_slice.append(slice1)
        slice2 = sequences[:,:, start_idx+2:start_idx+4]
        directions_slice.append(F.normalize(slice2))
        slice3 = sequences[:,:, start_idx+4].unsqueeze(2)
        length_slice.append(slice3)
    #[batch,seq,feature]
    centers_flat = torch.cat(centers_slice, dim=2)
    directions_flat = torch.cat(directions_slice, dim=2)
    length_flat = torch.cat(length_slice, dim=2)
    #[batch,seq_sum]
    length_greater_than_zero = (length_flat > 0).sum(dim=2).float()
    return centers_flat,directions_flat,length_greater_than_zero
def splitSequences2(sequences):
    centers_slice = []
    directions_slice = []
    length_slice=[]
    for i in range(sequences.shape[2] // 5):
        start_idx = i * 5
        slice1 = sequences[:,:, start_idx:start_idx+2]
        centers_slice.append(slice1)
        slice2 = sequences[:,:, start_idx+2:start_idx+4]
        directions_slice.append(F.normalize(slice2))
        slice3 = sequences[:,:, start_idx+4].unsqueeze(2)
        length_slice.append(slice3)
    #[batch,seq,feature]
    #将方向向量转换为角度
    centers_flat = torch.cat(centers_slice, dim=2)
    directions_flat = torch.cat(directions_slice, dim=2)
    length_flat = torch.cat(length_slice, dim=2)
    length_greater_than_zero = (length_flat > 0).sum(dim=2).float()
    line_num=length_greater_than_zero.unsqueeze(-1)
    return centers_flat,directions_flat,length_flat,line_num

def splitSequences(sequences):
    centers_slice = []
    directions_slice = []
    length_slice=[]
    thetas=[]
    for i in range(sequences.shape[2] // 5):
        start_idx = i * 5
        slice1 = sequences[:,:, start_idx:start_idx+2]
        centers_slice.append(slice1)
        slice2 = sequences[:,:, start_idx+2:start_idx+4]
        #dx=sequences[:,:, start_idx+2:start_idx+3]
        #dy=sequences[:,:, start_idx+3:start_idx+4]
        #theta=torch.atan2(dx,dy)
        #thetas.append(theta)
        #print("theta:",theta.shape)
        directions_slice.append(F.normalize(slice2))
        slice3 = sequences[:,:, start_idx+4].unsqueeze(2)
        length_slice.append(slice3)
    #[batch,seq,feature]
    #将方向向量转换为角度
    centers_flat = torch.cat(centers_slice, dim=2)
    directions_flat = torch.cat(directions_slice, dim=2)
    length_flat = torch.cat(length_slice, dim=2)
    #thetas_flat = torch.cat(thetas, dim=2)
    #print(length_flat.shape,thetas_flat[0,0:2,:])
    return centers_flat,directions_flat,length_flat

def unionSequence(center,direction,length):
    feature_len=center.shape[2]+direction.shape[2]+length.shape[2]
    out=torch.zeros(center.shape[0],center.shape[1],feature_len)
    for i in range(feature_len//5):
        out[:,:,i*5:i*5+2]=center[:,:,i*2:i*2+2]
        out[:,:,i*5+2:i*5+4]=direction[:,:,i*2:i*2+2]
        out[:,:,i*5+4]=length[:,:,i]
    return out
#根据线段的中心点坐标，向量，长度计算线段的两个端点坐标
def calculate_endpoints(center, vector, length):
    x_center, y_center = center
    x_vector, y_vector = vector

    # 计算单位向量
    magnitude = (x_vector**2 + y_vector**2)**0.5
    x_unit = x_vector / magnitude
    y_unit = y_vector / magnitude

    # 计算端点坐标
    endpoint1 = (x_center + 0.5 * length * x_unit, y_center + 0.5 * length * y_unit)
    endpoint2 = (x_center - 0.5 * length * x_unit, y_center - 0.5 * length * y_unit)

    return endpoint1, endpoint2

def calculate_endpoints_from_center_direction_length(segCenter, segDirection, segLength, ResX, ResY):
    if np.isclose(segLength,0):
        return [0,0],[0,0]
    # 计算线段的方向向量的长度
    segDirection_magnitude = np.sqrt(segDirection[0]**2 + segDirection[1]**2)
    # 根据segLength计算线段方向向量的标量系数
    scalar = segLength / (2 * segDirection_magnitude)
    # 计算u1和u2的坐标
    endpoint1 = [int((segCenter[0] + scalar * segDirection[0])* ResX), int((segCenter[1] + scalar * segDirection[1])* ResY)]
    endpoint2 = [int((segCenter[0] - scalar * segDirection[0])* ResX), int((segCenter[1] - scalar * segDirection[1])* ResY)]
    return endpoint1, endpoint2

def calculate_endpoints_from_center_direction_length_batch(center_batch, direction_batch, length_batch, ResX, ResY):
    # 计算线段的方向向量的长度
    segDirection_magnitude_batch = torch.sqrt(direction_batch[:, 0] ** 2 + direction_batch[:, 1] ** 2)
    
    # 根据segLength计算线段方向向量的标量系数
    scalar_batch = length_batch / (2 * segDirection_magnitude_batch)
    
    # 计算u1和u2的坐标
    endpoint1_x = (center_batch[:, 0] + scalar_batch * direction_batch[:, 0]) * ResX
    endpoint1_y = (center_batch[:, 1] + scalar_batch * direction_batch[:, 1]) * ResY
    endpoint2_x = (center_batch[:, 0] - scalar_batch * direction_batch[:, 0]) * ResX
    endpoint2_y = (center_batch[:, 1] - scalar_batch * direction_batch[:, 1]) * ResY
    
    return torch.stack([torch.stack([endpoint1_x, endpoint1_y], dim=1),
                        torch.stack([endpoint2_x, endpoint2_y], dim=1)], dim=1)
def getPointfromTensor(lineTensor,ResX,ResY):
    p1=[int(lineTensor[0]*ResX),int(lineTensor[1]*ResY)]
    p2=[int(lineTensor[2]*ResX),int(lineTensor[3]*ResY)]
    return p1,p2

def convertFrameToImageByPoint(frame,ResX,ResY,value=1,thickness=1):
    image=np.zeros((ResY,ResX),dtype=np.uint8)
    for l in range(51):
           lp=frame[l*4:l*4+4]
           p1,p2=getPointfromTensor(lp,ResX,ResY)
           cv2.line(image,p1,p2,value,thickness)
    return image
def convertFrameToImage(frame,ResX,ResY,value=1,thickness=1):
    image=np.zeros((ResY,ResX),dtype=np.uint8)
    for l in range(51):
           lp=frame[l*5:l*5+5]
           p1,p2=calculate_endpoints_from_center_direction_length([lp[0],lp[1]], [lp[2],lp[3]], lp[4], ResX, ResY)
           if lp[4]>0:
               cv2.line(image,p1,p2,value,thickness)
    return image
def saveSegment(segCenter, segDirection, segLength, ResX, ResY,savePath=None):
    p1,p2=calculate_endpoints_from_center_direction_length(segCenter, segDirection, segLength, ResX, ResY)
    if savePath:
        image=np.zeros((ResY,ResX),dtype=np.uint8)
        cv2.line(image,p1,p2,255,1)
        cv2.imwrite(savePath,image)


def saveFrame(frame_i,ResX,ResY,savePath=None,thickness=1):
    if savePath:
        image=np.zeros((ResY,ResX),dtype=np.uint8)
        for l in range(51):
            lp=frame_i[l*5:l*5+5]
            p1,p2=calculate_endpoints_from_center_direction_length([lp[0],lp[1]], [lp[2],lp[3]], lp[4], ResX, ResY)
            cv2.line(image,p1,p2,255,thickness)
        cv2.imwrite(savePath,image)

def binary_cross_entropy_loss(input, target):
    epsilon = 1e-15  # 避免取对数时出现无限大
    loss = - (target * np.log(input + epsilon) + (1 - target) * np.log(1 - input + epsilon))
    return np.mean(loss)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets,sequence_length):
        predictions=predictions.detach().numpy()
        targets=targets.detach().numpy()
        batch_size=predictions.shape[0]
        loss=0
        MSE_criterion=nn.MSELoss()
        for bs in range(batch_size):
            for f in range(sequence_length):
                pred=convertFrameToImage(predictions[bs,f,:],64,64)
                target=convertFrameToImage(targets[bs,f,:],64,64)
                loss_i=MSE_criterion(pred,target)
                loss+=loss_i
        # 将累加的损失转换为张量并返回
        return torch.tensor(loss, requires_grad=True)
if __name__=="__main__":
    # 示例数据
    center = (2, 3)
    vector = (1, 1)
    length = 4

    # 计算端点坐标
    endpoint1, endpoint2 = calculate_endpoints(center, vector, length)

    print("Endpoint 1:", endpoint1)
    print("Endpoint 2:", endpoint2)
    # 示例数据
    segCenter = [0.5, 0.5]
    segDirection = [0.6, 0.8]
    segLength = 0.5
    ResX = 100
    ResY = 100

    # 计算端点坐标
    p1, p2= calculate_endpoints_from_center_direction_length(segCenter, segDirection, segLength, ResX, ResY)

    print("u1_x:", endpoint1, endpoint2)
    image = np.zeros((ResX,ResY), dtype=np.uint8)
    cv2.line(image, p1, p2, 255, 1)
    cv2.imwrite("./line.jpg",image)
