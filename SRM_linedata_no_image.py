#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       数据合成示例
@Date     :2023/10/17 15:36:25
@Author      :chenbaolin
@version      :1.0
'''
from UBG.unblocks import *
from VirtualCamera.utils import *
import os
from VirtualCamera.CalibCamera import Camera
from VirtualCamera.Plane import Plane,CirclePlane
import numpy as np
from datetime import datetime
import time
import shutil
#创建DFN空间


def generate(randStart,randEnd,PB_set,save_dir=None,Rand=False,step_size=0.5,dfn_region=[12,12,50]):
	MaxX,MaxY,MaxZ=dfn_region[0],dfn_region[1],dfn_region[2]
	if save_dir is None:
		save_dir="./"
	for seed in range(randStart,randEnd):
		randSeed=seed
		#np.random.seed(randSeed)#设置随机种子
		fileSaveDir=save_dir+str(randSeed)
		if os.path.exists(fileSaveDir):
			shutil.rmtree(fileSaveDir)
		os.makedirs(fileSaveDir)
		#创建DFN空间
		dfn = DFN()
		dfn.set_RandomSeed(randSeed)#设置随机数-则相同随机数下生成的模型是相同的
		dfn.set_RegionMaxCorner(dfn_region)
		dfn.add_FractureSet()#添加裂隙集
		dfn.add_VolumeMapping()#体积映射P30
		
		while dfn.volumesMapping[0].get_P30(0)<0.01:
			for PB in PB_set:
				dfn.fractureSets[0].add_BaecherFracture(PB[0], PB[1], PB[2], PB[3], PB[4], PB[5])
		#块体生成器
		generator = Generator()#块体生成器
		#dfn.export_DFNVtk(f"{save_dir}{randSeed}")#export dfn
		#准备相机
		camera=Camera(position=[MaxX/2,MaxY/2,5],LookAt=[MaxX/2,MaxY/2,1000])#初始化
		camera.frustumDepth=2
		camera.focalLength=20 #调整焦距
		camera.ResX=6000 #pixel
		camera.ResY=4000 #pixel
		camera.SensorWidth=22.3 #mm
		camera.SensorHeight=14.9 #mm
		camera.getFov()
		observeDist=11 #相机与开挖面距离保持固定距离,相机始终正对着开挖平面
		start=1.0#起点，也可以设置随机采样，进而确保尽可能覆盖
		cameraHeight=1.6 #相机高度
		targetZ=start
		newStepSize=0
		frameId=0
		while targetZ<MaxZ-2:
			camera.setPosition([MaxX/2,cameraHeight,targetZ-observeDist]) #在OXY平面上，相机与位于区域中心点上
			camera.setLookAt([MaxX/2,cameraHeight+observeDist*np.tan(np.deg2rad(12)),targetZ])
			camera.getRT()
			#目标平面
			tpCenter=[MaxX/2,MaxY/2,targetZ]
			tp1=[MaxX,MaxY]
			tp2=[MaxX,0]
			tp3=[0,0]
			tp4=[0,MaxY]
			tpNormal=[0,0,1]
			tPlane=Plane(tpCenter,tpNormal)
			#遍历每个生成的裂隙集及其裂隙面
			lines=[]
			boundImage=[[0,0],[camera.ResX,0],[camera.ResX,camera.ResY],[0,camera.ResY]]
			boundOfSRM=[tp1,tp2,tp3,tp4]
			for i in range(dfn.FractureSetsNum):
				fs=dfn.fractureSets[i]
				n=fs.get_FractureNum() #集合中的裂隙数量
				for k in range(n):
					f=fs.fractures[k]#获取裂隙面
					#计算裂隙面与目标平面的交点
					#print("id",f.id,"Center:",f.get_Center(),"normal:",f.get_UnitVector(),"radius:",f.get_BoundingSphereRadius(),"area:",f.get_Area())
					if not np.isnan(f.get_Area()):
						#这里只支持圆形裂隙Slice
						frac=CirclePlane(f.get_Center(),f.get_UnitVector(),f.get_BoundingSphereRadius())
						flag,p1,p2=tPlane.intersectCirplane(frac)
						if flag:
							lines.append([p1,p2])
							segments=camera.getIntersectionLineSegment(p1,p2,boundImage)
							realLineInBoundOfSRM=getILSInBound(p1,p2,boundOfSRM)#Slice
							if segments:
								#print("realLineInBoundOfSRM",p1,p2,realLineInBoundOfSRM)
								pr1=realLineInBoundOfSRM[0] if realLineInBoundOfSRM else [0,0]
								pr2=realLineInBoundOfSRM[1] if realLineInBoundOfSRM else [0,0]
								q1=segments[0]
								q2=segments[1]
								q1x,q1y=q1[0]/camera.ResX,q1[1]/camera.ResY
								q2x,q2y=q2[0]/camera.ResX,q2[1]/camera.ResY
								seg_center=[(q1x+q2x)/2,(q1y+q2y)/2]
								direction=[q1x-q2x,q1y-q2y]
								length=np.linalg.norm([direction[0],direction[1]])
								direction=normalize(direction)
								#frame data
								data=[]
								data+=[f.id]
								data+=f.get_Center()
								data+=f.get_UnitVector()
								data+=[f.get_BoundingSphereRadius()]
								data+=seg_center
								data+=direction.tolist()
								data+=[length]
								data+=[observeDist,targetZ,newStepSize]
								data+=[q1x,q1y,q2x,q2y,pr1[0],pr1[1],pr2[0],pr2[1]]
								DataWriter(fileSaveDir+"/"+str(targetZ)+".txt",data)
			if Rand:
				newStepSize=1+np.random.uniform(0,1) #平均间距1.5m，最小1，最大2
				#newStepSize=np.round(abs(np.random.normal(1)),2)#随机正态分布均值为1，方差为1
			else:
				newStepSize=step_size
			targetZ=np.round(newStepSize+targetZ,2)
			frameId+=1

from LineDataPrepare_no_image import Convert
import argparse
parser = argparse.ArgumentParser(description='SRM fracture traces genenrator')
parser.add_argument('--start_seed', type=int, default=0) #测试集从64开始
parser.add_argument('--group_size', type=int, default=16)
parser.add_argument('--sequence_length', type=int, default=16)
parser.add_argument('--step_size', type=float, default=0.5)
parser.add_argument('--only_clip', type=int, default=1)
parser.add_argument('--do_generate', type=int, default=0)
parser.add_argument('--DataSetDir', type=str, default="./DataSet/")
parser.add_argument('--FractureDir', type=str, default="./Fracture/")
args = parser.parse_args()
print(args)

#define fracture set
MaxX,MaxY,MaxZ=12,12,500
PB_set1=[60,40,25,'exp',10,2] #0-15
PB_set2=[210, 70, 20,'exp',10,2] #16-31
PB_set3=[300, 60, 15,'exp',10,2] #32-47
PBDict={
	"PBSet1":[PB_set1],
	"PBSet2":[PB_set2],
	"PBSet3":[PB_set3],
	"PBFusion123":[PB_set1,PB_set2,PB_set3]
}
dfn_region=[MaxX,MaxY,MaxZ]
dataType=["NRandStep","FixStep05"]
listKeys=list(PBDict.keys())
sequence_length=args.sequence_length
only_clip=args.only_clip
do_generate=args.do_generate
group_size=args.group_size
start_seed=args.start_seed
DataSetDir=args.DataSetDir
FractureDir=args.FractureDir
step_size=args.step_size
for dt in dataType:
	gen_root=FractureDir+dt+"/"
	for i in range(4):
		k=listKeys[i]
		d=PBDict[k]
		if do_generate:
			generate(start_seed+group_size*i,start_seed+group_size*(i+1),d,gen_root,dt!="FixStep05",step_size=step_size,dfn_region=dfn_region)#注释则不重复生成
	for i in range(4):
		Convert(start_seed+group_size*i,start_seed+group_size*(i+1),listKeys[i],gen_root,DataSetDir+dt+'/',sequence_length,only_clip)
	Convert(start_seed+0,start_seed+group_size*4,dataSetName="PBAll",dataRoot=gen_root,saveRoot=DataSetDir+dt+'/',sequence_length=sequence_length,only_clip=only_clip)
	Convert(start_seed+0,start_seed+group_size*3,dataSetName="PBUnion123",dataRoot=gen_root,saveRoot=DataSetDir+dt+'/',sequence_length=sequence_length,only_clip=only_clip)




