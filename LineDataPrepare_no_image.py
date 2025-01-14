#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       data prepare:
@Date     :2023/09/21 13:56:04
@Author      :chenbaolin
@version      :1.0
'''
import numpy as np
import os
def Convert(startR,endR,dataSetName,dataRoot=None,saveRoot=None,sequence_length=8,only_clip=True):
    '''
        sequence_length:input_length+predict_length (or label_length)
    '''
    if dataRoot is None:
        dataRoot="./"
    if saveRoot is None:
        saveRoot="./"
    
    os.makedirs(saveRoot+dataSetName,exist_ok=True)
    segment_center_direction_length_list=[]
    frame_positions=[]
    clips_index=[]
    start_index=0
    endpoints_list=[]
    endpoints3d_list=[]
    for i in range(startR,endR):
        folder_path=os.path.join(dataRoot,str(i))
        file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        file_list.sort(key=lambda x: float(x.split('.txt')[0]))  # 按文件名中的数字排序
        frame_num=len(file_list)
        for file_name in file_list:
            frame=[0]*255
            endpoints=[0]*204
            endPoints3D=[0]*(6*51)
            file_path = os.path.join(folder_path, file_name)
            pos=float(file_name.split(".txt")[0])#保存位置
            frame_positions.append(pos)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines)>51 or len(lines)==0:
                    print(file_path)
                    f.close()
                    os.remove(file_path)
                    frame_num-=1
                else:
                    row=0
                    if not only_clip:
                        for line in lines:
                            values = line.strip().split(',')
                            d=[float(val) for val in values[8:13]]
                            points=[float(val) for val in values[16:20]]
                            endpoints[4*row:4*row+4]=points #裂隙线在图像上的端点坐标
                            points3d=[float(values[14])]+[float(val) for val in values[20:22]]+[float(values[14])]+[float(val) for val in values[22:24]]
                            endPoints3D[6*row:6*row+6]=points3d #裂隙线在图像上的端点坐标
                            #实际坐标
                            frame[5*row:5*row+5] = d
                            row+=1
                        segment_center_direction_length_list.append(frame)
                        endpoints_list.append(endpoints)
                        endpoints3d_list.append(endPoints3D)
        dif = frame_num - sequence_length
        for i in range(dif):
            clips=[]
            clips.append(i+start_index)
            clips.append(i+sequence_length+start_index)
            clips_index.append(clips)
        start_index+=frame_num
    # 保存为.npy文件
    if only_clip:
        clips_array=np.array(clips_index,dtype=np.int32)
        np.save(saveRoot+dataSetName+f'/frame_clips_{sequence_length}-train.npy', clips_array)
        position_array=np.array(frame_positions,dtype=np.float32)
        np.save(saveRoot+dataSetName+'/frame_position-train.npy', position_array)
    else:
        segment_center_direction_length = np.array(segment_center_direction_length_list,dtype=np.float32)
        position_array=np.array(frame_positions,dtype=np.float32)
        clips_array=np.array(clips_index,dtype=np.int32)
        endpoints_list_array = np.array(endpoints_list,dtype=np.float32)
        endpoints3d_list_array = np.array(endpoints3d_list,dtype=np.float32)
        print(segment_center_direction_length.shape,position_array.shape,clips_array.shape,endpoints_list_array.shape)
        np.save(saveRoot+dataSetName+f'/frame_clips_{sequence_length}-train.npy', clips_array)
        np.save(saveRoot+dataSetName+'/fracture_center_direction_length-train', segment_center_direction_length)
        np.save(saveRoot+dataSetName+'/fracture_endpoints-train.npy', endpoints_list_array)
        np.save(saveRoot+dataSetName+'/frame_position-train.npy', position_array)
        np.save(saveRoot+dataSetName+'/frame_clips-train.npy', clips_array)
        np.save(saveRoot+dataSetName+'/fracture_endpoints3d-train.npy', endpoints3d_list_array)
    print(dataSetName,"数据已保存")