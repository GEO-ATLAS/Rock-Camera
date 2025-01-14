#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       二值图生成--虚拟相机拍摄
@Date     :2023/09/21 10:01:21
@Author      :chenbaolin
@version      :1.0
'''
import cv2
import numpy as np
Root="./DataSetSmall"
DataRoot="FixStep05"
thickness=1 #线段宽度
size=128 #图像尺寸
for dataSetName in ["PBAll","PBSet1","PBSet2","PBSet3","PBUnion123","PBFusion123"]:
    endPoints=np.load(f"{Root}/{DataRoot}/{dataSetName}/fracture_endpoints-train.npy")
    print(endPoints.shape)
    dilatedImages=[]
    for j in range(endPoints.shape[0]):
        frame1=endPoints[j,:]*size
        image=np.zeros((size,size),dtype=np.uint8)
        for i in range(51):
            p1x,p1y,p2x,p2y=frame1[i*4],frame1[i*4+1],frame1[i*4+2],frame1[i*4+3]
            if np.isclose(p1x,0) and np.isclose(p2x,0) and np.isclose(p1y,0) and np.isclose(p2y,0):
                continue
            else:
                line_image=cv2.line(image,(int(p1x),int(p1y)),(int(p2x),int(p2y)),1,thickness)
        dilatedImages.append(line_image)
    dilatedImages=np.array(dilatedImages)
    print(dilatedImages.shape)
    #cv2.imwrite("./"+dataSetName+"2px.jpg",line_image*255)
    np.save(f"{Root}/{DataRoot}/{dataSetName}/fracture_images_raw_{size}_{thickness}px.npy",dilatedImages)
    print(dataSetName,"convert done!")